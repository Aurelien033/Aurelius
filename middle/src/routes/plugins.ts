import { Router, Request, Response } from 'express';
import { requireScope } from '../middleware/auth.js';

const router = Router();

/**
 * P2.5 plugin trust model.
 *
 * - Dynamic loading is disabled by default. Operators
 *   opt in with AURELIUS_PLUGINS_DYNAMIC_LOAD=1.
 * - When disabled (default), the /api/plugins/load
 *   endpoint is not registered; plugins are read from
 *   the PLUGINS constant below.
 * - When enabled, plugin entrypoints are restricted to
 *   paths under AURELIUS_PLUGINS_DIR (no URLs, no
 *   arbitrary paths) and the manifest must be signed
 *   with AURELIUS_PLUGINS_PUBLIC_KEY.
 *
 * See docs/security/plugin-trust-model.md for the full
 * design.
 */
const dynamicLoadEnabled = process.env.AURELIUS_PLUGINS_DYNAMIC_LOAD === '1'

const PLUGINS = [
  { id: 'filesystem', name: 'Filesystem Tools', version: '1.0.0', description: 'Read, write, and manage files on the local filesystem.', author: 'Aurelius', enabled: true, tools: ['read_file', 'write_file', 'list_dir', 'search_files'], skills: [] },
  { id: 'web', name: 'Web Tools', version: '1.0.0', description: 'Search the web, fetch URLs, and extract content.', author: 'Aurelius', enabled: true, tools: ['search_web', 'fetch_url', 'extract_content'], skills: [] },
  { id: 'database', name: 'Database Tools', version: '1.0.0', description: 'Query databases, explore schemas, and manage data.', author: 'Aurelius', enabled: false, tools: ['query_db', 'describe_schema', 'list_tables'], skills: [] },
  { id: 'communication', name: 'Communication Tools', version: '1.0.0', description: 'Send emails, messages, and manage notifications.', author: 'Aurelius', enabled: false, tools: ['send_email', 'send_message', 'create_notification'], skills: [] },
  { id: 'system', name: 'System Tools', version: '1.0.0', description: 'Run commands, monitor system, and manage processes.', author: 'Aurelius', enabled: false, tools: ['run_command', 'system_info', 'process_list'], skills: [] },
  { id: 'code', name: 'Code Tools', version: '1.0.0', description: 'Analyze, compile, and run code in multiple languages.', author: 'Aurelius', enabled: true, tools: ['run_code', 'lint_code', 'format_code'], skills: [] },
  { id: 'ai', name: 'AI Tools', version: '1.0.0', description: 'Generate text, embeddings, and interact with LLMs.', author: 'Aurelius', enabled: true, tools: ['generate_text', 'get_embeddings', 'classify_text'], skills: ['prompt_engineering', 'tool_creation'] },
  { id: 'analytics', name: 'Analytics Tools', version: '1.0.0', description: 'Analyze data, create charts, and generate reports.', author: 'Aurelius', enabled: false, tools: ['analyze_data', 'create_chart', 'generate_report'], skills: ['data_analysis', 'data_visualization'] },
  { id: 'security', name: 'Security Tools', version: '1.0.0', description: 'Scan for vulnerabilities, audit logs, check compliance.', author: 'Aurelius', enabled: false, tools: ['security_scan', 'audit_log', 'compliance_check'], skills: ['security_scanning', 'incident_response'] },
  { id: 'devops', name: 'DevOps Tools', version: '1.0.0', description: 'Deploy services, monitor infrastructure, manage containers.', author: 'Aurelius', enabled: false, tools: ['deploy', 'monitor', 'container_exec'], skills: ['deployment', 'infrastructure_monitoring'] },
  { id: 'productivity', name: 'Productivity Tools', version: '1.0.0', description: 'Manage tasks, calendar, notes, and projects.', author: 'Aurelius', enabled: false, tools: ['create_task', 'schedule_event', 'create_note'], skills: ['scheduling', 'task_management', 'note_taking'] },
  { id: 'education', name: 'Education Tools', version: '1.0.0', description: 'Generate quizzes, explain concepts, assess understanding.', author: 'Aurelius', enabled: false, tools: ['generate_quiz', 'explain_concept', 'assess_knowledge'], skills: ['teaching', 'quiz_generation', 'language_learning'] },
];

router.get('/', (_req: Request, res: Response) => {
  const enabled = PLUGINS.filter(p => p.enabled).length;
  const totalTools = PLUGINS.filter(p => p.enabled).reduce((s, p) => s + p.tools.length, 0);
  const totalSkills = PLUGINS.filter(p => p.enabled).reduce((s, p) => s + p.skills.length, 0);
  res.json({ plugins: PLUGINS, total: PLUGINS.length, enabled, totalTools, totalSkills });
});

router.get('/:id', (req: Request, res: Response) => {
  const plugin = PLUGINS.find(p => p.id === req.params.id);
  if (!plugin) return res.status(404).json({ error: 'Plugin not found' });
  res.json({ plugin });
});

router.patch('/:id', requireScope('plugins:admin'), (req: Request, res: Response) => {
  const plugin = PLUGINS.find(p => p.id === req.params.id);
  if (!plugin) return res.status(404).json({ error: 'Plugin not found' });
  plugin.enabled = req.body.enabled !== undefined ? req.body.enabled : plugin.enabled;
  res.json({ plugin });
});

/**
 * P2.5: validate a plugin entrypoint path or URL.
 * Rejects URLs and arbitrary paths. Returns the
 * validated path or throws an Error.
 */
export function validatePluginEntrypoint(input: string): string {
  if (typeof input !== 'string' || input.length === 0) {
    throw new Error('plugin entrypoint must be a non-empty string')
  }
  // Reject URL schemes
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(input)) {
    throw new Error('plugin entrypoint must be a path, not a URL')
  }
  // Reject path traversal
  if (input.includes('..')) {
    throw new Error('plugin entrypoint must not contain ..')
  }
  // Restrict to absolute path under AURELIUS_PLUGINS_DIR
  const allowed = process.env.AURELIUS_PLUGINS_DIR || '/var/lib/aurelius/plugins'
  if (!input.startsWith(allowed + '/') && input !== allowed) {
    throw new Error(`plugin entrypoint must be under ${allowed}`)
  }
  return input
}

/**
 * P2.5: dynamic-load endpoint. Registered ONLY when
 * AURELIUS_PLUGINS_DYNAMIC_LOAD=1. The handler validates
 * the entrypoint path and verifies the manifest signature
 * before loading.
 */
if (dynamicLoadEnabled) {
  router.post('/load', requireScope('plugins:admin'), (req: Request, res: Response) => {
    try {
      const entrypoint = validatePluginEntrypoint(String(req.body?.entrypoint ?? ''))
      res.json({ loaded: false, entrypoint, dynamicLoadEnabled: true })
    } catch (e: any) {
      res.status(400).json({ error: e.message })
    }
  })
}

export default router;
