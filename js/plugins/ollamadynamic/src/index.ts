import { Genkit } from 'genkit';
import { logger } from 'genkit/logging';
import {
  GenerateRequest,
  GenerateResponseData,
  GenerationCommonConfigSchema,
  getBasicUsageStats,
  MessageData,
} from 'genkit/model';
import { GenkitPlugin, genkitPlugin } from 'genkit/plugin';
import {
  ApiType,
  DynamicConfiguration,
  type OllamaDynamicPluginParams,
} from './types.js';

export { type OllamaDynamicPluginParams };

export function ollamadynamic(params: OllamaDynamicPluginParams): GenkitPlugin {
  return genkitPlugin('ollama', async (ai: Genkit) => {
    ollamadynamicModel(ai, params.dynamicConfig)
  });
}

function ollamadynamicModel(
  ai: Genkit,
  dynamicConfig: DynamicConfiguration
) {
  return ai.defineModel(
    {
      name: `ollamadynamic/model`,
      label: `Ollama Dynamic`,
      configSchema: GenerationCommonConfigSchema,
      supports: {
        multiturn: true,
        media: true,
        tools: false,
        systemRole: false,
      },
    },
    async (input, streamingCallback) => {
      const options: Record<string, any> = {};

      let requestId:string = "";
      let modelName:string = "";
      // @ts-ignore
      if (input.context?.[0]?.metadata) {
        // @ts-ignore
        requestId = input.context?.[0]?.metadata["requestId"];
        // @ts-ignore
        modelName = input.context?.[0]?.metadata["modelName"];
      }
      // @ts-ignore
      logger.debug(`ollama dynamic context (${JSON.stringify( input?.context?.[0])})`);

      const config = await dynamicConfig({modelName,requestId},input) as { serverAddress: string; name: string; type?: ApiType; authToken:Record<string, string>};

      let model = {name:config.name, type:config.type};
      let requestHeaders = config.authToken;
      let serverAddress = config.serverAddress;
      
      if (input.config?.temperature !== undefined) {
        options.temperature = input.config.temperature;
      }
      if (input.config?.topP !== undefined) {
        options.top_p = input.config.topP;
      }
      if (input.config?.topK !== undefined) {
        options.top_k = input.config.topK;
      }
      if (input.config?.stopSequences !== undefined) {
        options.stop = input.config.stopSequences.join('');
      }
      if (input.config?.maxOutputTokens !== undefined) {
        options.num_predict = input.config.maxOutputTokens;
      }
      const type = model.type ?? 'chat';
      
      const request = toOllamadynamicRequest(
        model.name,
        input,
        options,
        type,
        !!streamingCallback
      );
      logger.debug(request, `ollama request (${type})`);

      const extraHeaders = requestHeaders
        ? requestHeaders
        : {};

      let res;
      try {
        res = await fetch(
          serverAddress + (type === 'chat' ? '/api/chat' : '/api/generate'),
          {
            method: 'POST',
            body: JSON.stringify(request),
            headers: {
              'Content-Type': 'application/json',
              ...extraHeaders,
            },
          }
        );
      } catch (e) {
        const cause = (e as any).cause;
        if (
          cause &&
          cause instanceof Error &&
          cause.message?.includes('ECONNREFUSED')
        ) {
          cause.message += '. Make sure the Ollama server is running.';
          throw cause;
        }
        throw e;
      }
      if (!res.body) {
        throw new Error('Response has no body');
      }

      let message: MessageData;

      if (streamingCallback) {
        const reader = res.body.getReader();
        const textDecoder = new TextDecoder();
        let textResponse = '';
        for await (const chunk of readChunks(reader)) {
          const chunkText = textDecoder.decode(chunk);
          const json = JSON.parse(chunkText);
          const message = parseMessage(json, type);
          streamingCallback({
            index: 0,
            content: message.content,
          });
          textResponse += message.content[0].text;
        }
        message = {
          role: 'model',
          content: [
            {
              text: textResponse,
            },
          ],
        };
      } else {
        const txtBody = await res.text();
        const json = JSON.parse(txtBody);
        logger.debug(txtBody, 'ollama raw response');

        message = parseMessage(json, type);
      }

      return {
        message,
        usage: getBasicUsageStats(input.messages, message),
        finishReason: 'stop',
      } as GenerateResponseData;
    }
  );
}

function parseMessage(response: any, type: ApiType): MessageData {
  if (response.error) {
    throw new Error(response.error);
  }
  if (type === 'chat') {
    return {
      role: toGenkitRole(response.message.role),
      content: [
        {
          text: response.message.content,
        },
      ],
    };
  } else {
    return {
      role: 'model',
      content: [
        {
          text: response.response,
        },
      ],
    };
  }
}

function toOllamadynamicRequest(
  name: string,
  input: GenerateRequest,
  options: Record<string, any>,
  type: ApiType,
  stream: boolean
) {
  const request: any = {
    model: name,
    options,
    stream,
  };
  if (type === 'chat') {
    const messages: Message[] = [];
    input.messages.forEach((m) => {
      let messageText = '';
      const images: string[] = [];
      m.content.forEach((c) => {
        if (c.text) {
          messageText += c.text;
        }
        if (c.media) {
          images.push(c.media.url);
        }
      });
      messages.push({
        role: toOllamaRole(m.role),
        content: messageText,
        images: images.length > 0 ? images : undefined,
      });
    });
    request.messages = messages;
  } else {
    request.prompt = getPrompt(input);
    request.system = getSystemMessage(input);
  }
  return request;
}

function toOllamaRole(role) {
  if (role === 'model') {
    return 'assistant';
  }
  return role; // everything else seems to match
}

function toGenkitRole(role) {
  if (role === 'assistant') {
    return 'model';
  }
  return role; // everything else seems to match
}

function readChunks(reader) {
  return {
    async *[Symbol.asyncIterator]() {
      let readResult = await reader.read();
      while (!readResult.done) {
        yield readResult.value;
        readResult = await reader.read();
      }
    },
  };
}

function getPrompt(input: GenerateRequest): string {
  return input.messages
    .filter((m) => m.role !== 'system')
    .map((m) => m.content.map((c) => c.text).join())
    .join();
}

function getSystemMessage(input: GenerateRequest): string {
  return input.messages
    .filter((m) => m.role === 'system')
    .map((m) => m.content.map((c) => c.text).join())
    .join();
}

interface Message {
  role: string;
  content: string;
  images?: string[];
}
