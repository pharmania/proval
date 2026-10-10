import { describe, expect, it, mock, spyOn } from "bun:test";
import { resolve } from "node:path";

// Keep the module mocks below isolated from the other API tests
if (process.env.PROVAL_ACTIVITY_RETRY_TEST_CHILD !== "1") {
    it("passes activity retry integration in an isolated process", () => {
        const result = Bun.spawnSync([process.execPath, "test", import.meta.path], {
            env: { ...process.env, PROVAL_ACTIVITY_RETRY_TEST_CHILD: "1", DB_FILE_NAME: ":memory:" },
            stdout: "pipe",
            stderr: "pipe",
        });
        if (result.exitCode !== 0) {
            throw new Error(result.stdout.toString() + result.stderr.toString());
        }
        expect(result.exitCode).toBe(0);
    }, 30000);
} else {
    process.env.DB_FILE_NAME = ":memory:";
    process.env.ENCRYPTION_KEY = Buffer.alloc(32, 13).toString("base64");

    const usage = { inputToken: 1, cachedInputToken: 0, outputToken: 1 };
    const runPullRequestReviewMock = mock(async () => usage);
    const runPullRequestReplyMock = mock(async () => usage);
    const runIssueReplyOnOpenMock = mock(async () => usage);
    const runIssueReplyMock = mock(async () => usage);

    mock.module("../../agent/pull-request/index.js", () => ({
        runPullRequestReview: runPullRequestReviewMock,
        runPullRequestReply: runPullRequestReplyMock,
    }));
    mock.module("../../agent/issue/index.js", () => ({
        runIssueReplyOnOpen: runIssueReplyOnOpenMock,
        runIssueReply: runIssueReplyMock,
    }));

    const { and, desc, eq, gt } = await import("drizzle-orm");
    const { migrate } = await import("drizzle-orm/bun-sqlite/migrator");
    const { default: db } = await import("../../db/index.js");
    const { activityTable, gitProviderAccessTable, modelProviderTable, repositoryTable } = await import("@proval/db");
    const { encrypt } = await import("../../util/encrypt.js");
    const { ActivityService } = await import("./activity.service.js");

    migrate(db, { migrationsFolder: resolve(import.meta.dir, "../../../../../packages/db/src/migration") });

    const consoleError = spyOn(console, "error").mockImplementation(() => {});

    const [modelProvider] = await db
        .insert(modelProviderTable)
        .values({
            provider: "openai",
            label: "Test",
            baseUrl: "http://localhost:1234/v1",
            apiKey: encrypt("test key"),
        })
        .returning({ id: modelProviderTable.id });
    const [access] = await db
        .insert(gitProviderAccessTable)
        .values({
            provider: "gitlab",
            name: "Test",
            baseUrl: "https://gitlab.example.com",
            accessToken: encrypt("glpat-test"),
        })
        .returning({ id: gitProviderAccessTable.id });
    const [repository] = await db
        .insert(repositoryTable)
        .values({
            path: "group/repo",
            provider: "gitlab",
            gitProviderAccessId: access.id,
            gitProviderRepositoryId: 5,
            accessToken: encrypt("glpat-repository"),
            modelProviderId: modelProvider.id,
            modelName: "test-model",
        })
        .returning({ id: repositoryTable.id });

    const service = new ActivityService();

    async function insertFailedActivity(values: {
        type: "pr_review" | "pr_reply" | "issue_open" | "issue_reply";
        targetIid: number;
        targetCommentId?: number | null;
        targetInlineReviewId?: string | null;
    }): Promise<number> {
        const [activity] = await db
            .insert(activityTable)
            .values({
                repositoryId: repository.id,
                repositoryPath: "group/repo",
                provider: "gitlab",
                modelProviderId: modelProvider.id,
                modelName: "test-model",
                type: values.type,
                status: "failed",
                targetIid: values.targetIid,
                targetCommentId: values.targetCommentId ?? null,
                targetInlineReviewId: values.targetInlineReviewId ?? null,
                logVersion: "1",
                errorMessage: "boom",
            })
            .returning({ id: activityTable.id });
        return activity.id;
    }

    async function waitForRetriedActivity(previousId: number) {
        const deadline = Date.now() + 3000;
        while (Date.now() < deadline) {
            const [activity] = await db
                .select()
                .from(activityTable)
                .where(and(eq(activityTable.repositoryId, repository.id), gt(activityTable.id, previousId)))
                .orderBy(desc(activityTable.id))
                .limit(1);
            if (activity?.status === "completed") {
                return activity;
            }
            await Bun.sleep(10);
        }
        throw new Error("Timed out waiting for the retried activity");
    }

    describe("activity retry", () => {
        it("replies to the same pull request comment", async () => {
            const failedId = await insertFailedActivity({
                type: "pr_reply",
                targetIid: 7,
                targetCommentId: 4242,
                targetInlineReviewId: "discussion-1",
            });

            await service.retry(failedId);

            const retried = await waitForRetriedActivity(failedId);
            expect(runPullRequestReplyMock).toHaveBeenCalledWith(
                expect.objectContaining({ prIid: 7, commentId: 4242, inlineReviewId: "discussion-1" }),
            );
            expect(retried.targetCommentId).toBe(4242);
            expect(retried.targetInlineReviewId).toBe("discussion-1");
        });

        it("replies to the same issue comment", async () => {
            const failedId = await insertFailedActivity({
                type: "issue_reply",
                targetIid: 11,
                targetCommentId: 5151,
            });

            await service.retry(failedId);

            const retried = await waitForRetriedActivity(failedId);
            expect(runIssueReplyMock).toHaveBeenCalledWith(expect.objectContaining({ issueIid: 11, commentId: 5151 }));
            expect(retried.targetCommentId).toBe(5151);
        });

        it("still runs the issue open workflow", async () => {
            const failedId = await insertFailedActivity({ type: "issue_open", targetIid: 13 });

            await service.retry(failedId);

            const retried = await waitForRetriedActivity(failedId);
            expect(runIssueReplyOnOpenMock).toHaveBeenCalledWith(expect.objectContaining({ issueIid: 13 }));
            expect(retried.type).toBe("issue_open");
        });

        it("refuses a reply without a stored comment", async () => {
            const callCount = runPullRequestReplyMock.mock.calls.length;
            const failedId = await insertFailedActivity({ type: "pr_reply", targetIid: 7 });

            await expect(service.retry(failedId)).rejects.toThrow("This activity has no comment to reply to");
            expect(runPullRequestReplyMock.mock.calls.length).toBe(callCount);
        });
    });

    consoleError.mockRestore();
}
