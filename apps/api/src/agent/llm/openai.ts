import OpenAI from "openai";
import type { LlmSender, Message } from "./loop.js";
import type { SenderSDKConfig } from "./factory.js";

export function createOpenAiSender(config: SenderSDKConfig): LlmSender {
    const client = new OpenAI({
        apiKey: config.apiKey,
        baseURL: config.baseURL,
        timeout: config.timeoutSecond * 1000,
        fetch: config.fetch,
    });

    return {
        async send(messages, tools) {
            const openAiTools: OpenAI.Chat.ChatCompletionTool[] = tools.map((t) => ({
                type: "function",
                function: {
                    name: t.name,
                    description: t.description,
                    parameters: t.parameters,
                },
            }));

            // Stream the completion. A buffered response stays silent while the
            // model reasons, so an edge proxy in front of the model (Cloudflare
            // answers a silent origin with 524 after 100 s) can cut the call
            // before the model finishes. Streaming sends the first bytes
            // immediately and keeps the connection alive.
            const stream = await client.chat.completions.create({
                model: config.model,
                messages: messages.map(convertToOpenAiMessage),
                ...(openAiTools.length > 0 ? { tools: openAiTools, tool_choice: "auto" as const } : {}),
                ...(config.reasoningEffort
                    ? {
                          reasoning_effort: config.reasoningEffort as OpenAI.Chat.ChatCompletionCreateParams["reasoning_effort"],
                      }
                    : {}),
                stream: true,
                stream_options: { include_usage: true },
            });

            const accumulated = await accumulateChatCompletionStream(stream);

            // A provider can answer 200 and report the failure inside the stream.
            if (accumulated.error) {
                throw new Error(accumulated.error);
            }

            return {
                message: {
                    role: "assistant",
                    content: accumulated.content,
                    toolCalls: accumulated.toolCalls,
                },
                finishReason: accumulated.finishReason ?? "stop",
                requestId: accumulated.id,
                usage: {
                    inputToken: accumulated.promptTokens,
                    outputToken: accumulated.completionTokens,
                    cachedInputToken: accumulated.cachedInputTokens,
                },
            };
        },
        getModel() {
            return { model: config.model, provider: "openai", baseUrl: config.baseURL };
        },
    };
}

export type AccumulatedChatCompletion = {
    id: string | null;
    content: string | null;
    toolCalls: { id: string; name: string; arguments: string }[] | undefined;
    finishReason: OpenAI.Chat.ChatCompletion.Choice["finish_reason"] | null;
    promptTokens: number;
    completionTokens: number;
    cachedInputTokens: number;
    error: string | null;
};

// Rebuild one completion from the stream chunks: content, tool-call fragments
// and usage. Tool calls arrive in pieces, indexed by `tool_calls[].index`, and
// the arguments of one call can be split across several chunks.
export async function accumulateChatCompletionStream(
    stream: AsyncIterable<OpenAI.Chat.ChatCompletionChunk>,
): Promise<AccumulatedChatCompletion> {
    let id: string | null = null;
    let content = "";
    let finishReason: OpenAI.Chat.ChatCompletion.Choice["finish_reason"] | null = null;
    let promptTokens = 0;
    let completionTokens = 0;
    let cachedInputTokens = 0;
    let error: string | null = null;
    const toolCalls = new Map<number, { id: string; name: string; arguments: string }>();

    for await (const chunk of stream) {
        const chunkError = (chunk as { error?: { message?: string } }).error;
        if (chunkError) {
            error = chunkError.message ?? "The model provider returned an error";
            continue;
        }

        if (chunk.id) id = chunk.id;

        if (chunk.usage) {
            promptTokens = chunk.usage.prompt_tokens ?? promptTokens;
            completionTokens = chunk.usage.completion_tokens ?? completionTokens;
            cachedInputTokens = chunk.usage.prompt_tokens_details?.cached_tokens ?? cachedInputTokens;
        }

        const choice = chunk.choices?.[0];
        if (!choice) continue;
        if (choice.finish_reason) finishReason = choice.finish_reason;

        const delta = choice.delta;
        if (!delta) continue;
        if (delta.content) content += delta.content;

        for (const toolCall of delta.tool_calls ?? []) {
            const index = toolCall.index ?? 0;
            const current = toolCalls.get(index) ?? { id: "", name: "", arguments: "" };
            if (toolCall.id) current.id = toolCall.id;
            if (toolCall.function?.name) current.name = toolCall.function.name;
            if (toolCall.function?.arguments) current.arguments += toolCall.function.arguments;
            toolCalls.set(index, current);
        }
    }

    return {
        id,
        content: content.length > 0 ? content : null,
        toolCalls:
            toolCalls.size > 0
                ? [...toolCalls.entries()].sort(([a], [b]) => a - b).map(([, value]) => value)
                : undefined,
        finishReason,
        promptTokens,
        completionTokens,
        cachedInputTokens,
        error,
    };
}

function convertToOpenAiMessage(m: Message): OpenAI.Chat.ChatCompletionMessageParam {
    switch (m.role) {
        case "system":
            return { role: "system", content: m.content ?? "" };
        case "user":
            return { role: "user", content: m.content ?? "" };
        case "assistant":
            if (m.toolCalls && m.toolCalls.length > 0) {
                return {
                    role: "assistant",
                    content: m.content ?? null,
                    tool_calls: m.toolCalls.map((tc) => ({
                        id: tc.id,
                        type: "function",
                        function: { name: tc.name, arguments: tc.arguments },
                    })),
                };
            }
            return { role: "assistant", content: m.content ?? null };
        case "tool":
            return {
                role: "tool",
                tool_call_id: m.toolCallId ?? "",
                content: m.content ?? "",
            };
    }
}
