
import Fastify from "fastify";
import { OpenRouter } from "@openrouter/sdk";

const app = Fastify({ logger: true });

const apiKey = process.env.OPENROUTER_API_KEY;
const model = process.env.OPENROUTER_MODEL ?? "recursive/flint-v1.0:free";
const port = Number(process.env.PORT ?? 3000);

if (!apiKey) {
  throw new Error("OPENROUTER_API_KEY is missing from .env");
}

if (!Number.isInteger(port) || port < 1 || port > 65535) {
  throw new Error("PORT must be a valid port number");
}

const openrouter = new OpenRouter({ apiKey });

type ChatBody = {
  question: string;
};

const chatSchema = {
  body: {
    type: "object",
    required: ["question"],
    additionalProperties: false,
    properties: {
      question: {
        type: "string",
        minLength: 1,
        maxLength: 2000,
      },
    },
  },
};

app.get("/health", async () => {
  return { status: "ok" };
});

// Non-streaming endpoint
app.post<{ Body: ChatBody }>(
  "/ai/chat",
  { schema: chatSchema },
  async (request, reply) => {
    const question = request.body.question.trim();

    if (!question) {
      return reply.code(400).send({
        error: "Question cannot be empty",
      });
    }

    try {
      const response = await openrouter.chat.send({
        chatRequest: {
          model,
          messages: [
            {
              role: "system",
              content: "You are a helpful programming assistant.",
            },
            {
              role: "user",
              content: question,
            },
          ],
          stream: false,
        },
      });

      if (response instanceof ReadableStream) {
        return reply.code(502).send({
          error: "Expected a non-streaming response",
        });
      }

      const answer = response.choices[0]?.message.content;

      if (typeof answer !== "string") {
        request.log.error(
          { response },
          "OpenRouter returned no text answer",
        );

        return reply.code(502).send({
          error: "AI service returned no text answer",
        });
      }

      return { answer };
    } catch (error) {
      request.log.error(
        { err: error },
        "OpenRouter chat request failed",
      );

      return reply.code(502).send({
        error: "AI service request failed",
      });
    }
  },
);

// Streaming endpoint using Server-Sent Events (SSE)
app.post<{ Body: ChatBody }>(
  "/ai/chat/stream",
  { schema: chatSchema },
  async (request, reply) => {
    const question = request.body.question.trim();

    if (!question) {
      return reply.code(400).send({
        error: "Question cannot be empty",
      });
    }

    const controller = new AbortController();

    try {
      const stream = await openrouter.chat.send(
        {
          chatRequest: {
            model,
            messages: [
              {
                role: "system",
                content: "You are a helpful programming assistant.",
              },
              {
                role: "user",
                content: question,
              },
            ],
            stream: true,
          },
        },
        { signal: controller.signal },
      );

      if (!(stream instanceof ReadableStream)) {
        return reply.code(502).send({
          error: "Expected a streaming response",
        });
      }

      // We manage the raw response because SSE needs incremental writes.
      reply.hijack();

      reply.raw.writeHead(200, {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
      });

      reply.raw.flushHeaders();

      let clientDisconnected = false;

      reply.raw.on("close", () => {
        if (!reply.raw.writableEnded) {
          clientDisconnected = true;
          controller.abort();
        }
      });

      try {
        for await (const chunk of stream) {
          if (clientDisconnected) {
            break;
          }

          if ("error" in chunk) {
            throw new Error("OpenRouter stream reported an error");
          }

          const content = chunk.choices[0]?.delta?.content;

          if (typeof content === "string" && content.length > 0) {
            const event = JSON.stringify({ text: content });
            reply.raw.write(`data: ${event}\n\n`);
          }

          // Usage is logged on the server, not streamed to the client.
          if (chunk.usage) {
            request.log.info(
              { usage: chunk.usage },
              "OpenRouter usage",
            );
          }
        }

        if (!clientDisconnected && !reply.raw.destroyed) {
          reply.raw.write("data: [DONE]\n\n");
          reply.raw.end();
        }
      } catch (error) {
        if (clientDisconnected || controller.signal.aborted) {
          request.log.debug("Streaming client disconnected");
          return;
        }

        request.log.error(
          { err: error },
          "OpenRouter stream failed",
        );

        // Headers are already sent, so the HTTP status cannot be changed.
        if (!reply.raw.destroyed && !reply.raw.writableEnded) {
          reply.raw.write(
            `data: ${JSON.stringify({ error: "AI stream failed" })}\n\n`,
          );
          reply.raw.end();
        }
      }
    } catch (error) {
      request.log.error(
        { err: error },
        "Unable to start OpenRouter stream",
      );

      if (!reply.sent) {
        return reply.code(502).send({
          error: "AI service request failed",
        });
      }
    }
  },
);


type Message = {
  role: "user" | "assistant";
  content: string;
};

const conversations = new Map<string, Message[]>();

// Create a new conversation
app.post("/ai/conversations", async (_request, reply) => {
  const conversationId = crypto.randomUUID();

  conversations.set(conversationId, []);

  return reply.code(201).send({ conversationId });
});

// Send a message within an existing conversation
app.post<{
  Params: { conversationId: string };
  Body: { question: string };
}>(
  "/ai/conversations/:conversationId/chat",
  {
    schema: {
      params: {
        type: "object",
        required: ["conversationId"],
        properties: {
          conversationId: { type: "string", format: "uuid" },
        },
      },
      body: {
        type: "object",
        required: ["question"],
        additionalProperties: false,
        properties: {
          question: {
            type: "string",
            minLength: 1,
            maxLength: 2000,
          },
        },
      },
    },
  },
  async (request, reply) => {
    const { conversationId } = request.params;
    const question = request.body.question.trim();

    if (!question) {
      return reply.code(400).send({
        error: "Question cannot be empty",
      });
    }

    const history = conversations.get(conversationId);

    if (!history) {
      return reply.code(404).send({
        error: "Conversation not found",
      });
    }

    // Keep the most recent 20 messages (10 user/assistant turns).
    const recentHistory = history.slice(-20);

    try {
      const response = await openrouter.chat.send({
        chatRequest: {
          model,
          messages: [
            {
              role: "system",
              content: "You are a helpful programming assistant.",
            },
            ...recentHistory,
            {
              role: "user",
              content: question,
            },
          ],
          stream: false,
        },
      });

      if (response instanceof ReadableStream) {
        return reply.code(502).send({
          error: "Expected a non-streaming response",
        });
      }

      const answer = response.choices[0]?.message.content;

      if (typeof answer !== "string") {
        return reply.code(502).send({
          error: "AI service returned no text answer",
        });
      }

      // Save both sides only after a successful response.
      const updatedHistory: Message[] = [
        ...recentHistory,
        { role: "user", content: question },
        { role: "assistant", content: answer },
      ];

      conversations.set(conversationId, updatedHistory);

      return {
        conversationId,
        answer,
        messageCount: updatedHistory.length,
      };
    } catch (error) {
      request.log.error(
        { err: error },
        "Conversation request failed",
      );

      return reply.code(502).send({
        error: "AI service request failed",
      });
    }
  },
);

app.addHook("onClose", async () => {
  app.log.info("Fastify server shutting down");
});

try {
  await app.listen({
    port,
    host: "127.0.0.1",
  });
} catch (error) {
  app.log.error({ err: error }, "Failed to start server");
  process.exit(1);
}
