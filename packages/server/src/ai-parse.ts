import Anthropic from '@anthropic-ai/sdk';
import { PARSE_SYSTEM_PROMPT } from './prompts/parse.js';
import { PARSE_RESPONSE_SCHEMA } from './prompts/parse-schema.js';

// Lazy client init — server should boot fine without ANTHROPIC_API_KEY;
// only the /api/ai/parse route fails when the key is missing.
let client: Anthropic | null = null;
function getClient(): Anthropic {
  if (client) return client;
  if (!process.env.ANTHROPIC_API_KEY) {
    throw new Error('ANTHROPIC_API_KEY is not set');
  }
  client = new Anthropic();
  return client;
}

// ─── Request / response types (mirror the client's expectations) ───

export interface AIParseRequest {
  input: string;
  availableFields: string[];
  existingRules: string[];
  existingColumns: string[];
}

type Operator = 'gt' | 'gte' | 'lt' | 'lte' | 'eq' | 'neq';

interface ConditionLhs {
  kind: 'field' | 'expr';
  field?: string;
  expression?: string;
}
interface ConditionRhs {
  kind: 'literal' | 'field' | 'expr';
  value?: number;
  field?: string;
  expression?: string;
}
interface Condition {
  lhs: ConditionLhs;
  operator: Operator;
  rhs: ConditionRhs;
}
interface ParsedRule {
  name: string;
  description: string;
  logic: 'AND' | 'OR';
  scope: 'call' | 'put' | 'row';
  conditions: Condition[];
}
interface ParsedColumn {
  name: string;
  expression: string;
  format: { type: 'number' | 'percentage' | 'currency'; decimals: number };
}
interface AmbiguousOption {
  label: string;
  intent: 'rule' | 'column';
  description: string;
}

export interface AIParseResponse {
  intent: 'rule' | 'column' | 'ambiguous';
  humanReadable: string;
  // 0..1 — how confident the model is in its parse. Coarse signal driving
  // the green/yellow/red confidence bar and the "best guess" warning banner.
  confidence: number;
  rule?: ParsedRule;
  column?: ParsedColumn;
  options?: AmbiguousOption[];
}

function buildUserMessage(req: AIParseRequest): string {
  const lines = [`User input: "${req.input}"`];
  if (req.existingRules.length) {
    lines.push(`Active rules (avoid exact-name duplication): ${req.existingRules.join(', ')}`);
  }
  if (req.existingColumns.length) {
    lines.push(`Active columns (avoid exact-name duplication): ${req.existingColumns.join(', ')}`);
  }
  lines.push('Parse this and respond with the JSON object only.');
  return lines.join('\n');
}

export async function parseAi(req: AIParseRequest): Promise<AIParseResponse> {
  const c = getClient();
  const t0 = Date.now();

  const response = (await c.messages.create({
    model: 'claude-haiku-4-5',
    max_tokens: 768,
    system: PARSE_SYSTEM_PROMPT,
    messages: [{ role: 'user', content: buildUserMessage(req) }],
    // `output_config` is the GA structured-output knob; cast to bypass the
    // SDK's older typed params shape.
    ...({
      output_config: {
        format: {
          type: 'json_schema',
          schema: PARSE_RESPONSE_SCHEMA as unknown as Record<string, unknown>,
        },
      },
    } as Record<string, unknown>),
  } as Anthropic.MessageCreateParams)) as Anthropic.Message;

  // The structured-output path returns one text block whose text is the JSON.
  const block = response.content[0];
  if (block?.type !== 'text') {
    throw new Error('Empty response from model');
  }
  const parsed = JSON.parse(block.text) as AIParseResponse;

  // Clamp values that JSON Schema can't bound (structured outputs reject
  // numerical constraints).
  if (parsed.column?.format) {
    const d = parsed.column.format.decimals;
    parsed.column.format.decimals = Math.max(0, Math.min(6, Math.round(d)));
  }
  if (typeof parsed.confidence === 'number') {
    parsed.confidence = Math.max(0, Math.min(1, parsed.confidence));
  } else {
    parsed.confidence = 0.7;
  }

  const u = response.usage;
  console.log(
    `[ai/parse] ${Date.now() - t0}ms · in=${u.input_tokens} out=${u.output_tokens}`
    + (u.cache_read_input_tokens ? ` cache_read=${u.cache_read_input_tokens}` : '')
    + ` · "${req.input.slice(0, 60)}${req.input.length > 60 ? '…' : ''}"`,
  );
  return parsed;
}
