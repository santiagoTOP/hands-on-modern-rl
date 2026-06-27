/* global process */
import { defineConfig } from 'vitepress'
import { createLogger } from 'vite'
import { MermaidMarkdown } from 'vitepress-plugin-mermaid'
import { createRequire } from 'module'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const require = createRequire(import.meta.url)
const markdownItFootnote = require('markdown-it-footnote')
const markdownItContainer = require('markdown-it-container')
const katex = require('katex')

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const packageJsonPath = path.resolve(__dirname, '../../package.json')
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'))
const docsRoot = path.resolve(__dirname, '..')
const assetManifestPath = path.resolve(
  docsRoot,
  'public/optimized/asset-manifest.json'
)

function loadAssetManifest() {
  if (!fs.existsSync(assetManifestPath)) return { assets: {} }

  try {
    return JSON.parse(fs.readFileSync(assetManifestPath, 'utf8'))
  } catch {
    return { assets: {} }
  }
}

const assetManifest = loadAssetManifest()

const isVercel = process.env.VERCEL === '1' || !!process.env.VERCEL_URL

function parseRepository() {
  const repositoryUrl =
    process.env.GITHUB_REPOSITORY ||
    packageJson.repository?.url ||
    packageJson.repository ||
    ''

  if (repositoryUrl.includes('/')) {
    if (repositoryUrl.includes(':')) {
      const sshMatch = repositoryUrl.match(/github\.com:(.+?)\/(.+?)(\.git)?$/)
      if (sshMatch) {
        return { owner: sshMatch[1], repo: sshMatch[2] }
      }
    }

    const normalized = repositoryUrl
      .replace(/^https?:\/\/github\.com\//, '')
      .replace(/^git@github\.com:/, '')
      .replace(/\.git$/, '')

    if (normalized.includes('/')) {
      const [owner, repo] = normalized.split('/')
      return { owner, repo }
    }
  }

  return { owner: 'walkinglabs', repo: packageJson.name || 'course-template' }
}

const { owner, repo } = parseRepository()
const base = process.env.BASE || (isVercel ? '/' : `/${repo}/`)
const siteUrl = process.env.SITE_URL || `https://${owner}.github.io/${repo}`
const editLinkPattern = `https://github.com/${owner}/${repo}/edit/main/docs/:path`
const enableLocalSearch = process.env.LOCAL_SEARCH !== '0'
const mermaidConfig = {
  securityLevel: 'loose',
  startOnLoad: false
}

function mermaidConfigPlugin() {
  const virtualModuleId = 'virtual:mermaid-config'
  const resolvedVirtualModuleId = `\0${virtualModuleId}`

  return {
    name: 'local-mermaid-config',
    resolveId(id) {
      if (id === virtualModuleId) {
        return resolvedVirtualModuleId
      }
    },
    load(id) {
      if (id === resolvedVirtualModuleId) {
        return `export default ${JSON.stringify(mermaidConfig)}`
      }
    }
  }
}

function normalizeBrokenDocPathPlugin() {
  const canonicalSegments = ['appendix_math', 'linear-algebra-basics']

  return {
    name: 'normalize-broken-doc-path',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url) return next()

        const url = new URL(req.url, 'http://localhost')
        if (!url.pathname.includes('appendix')) return next()

        const decodedPathname = decodeURIComponent(url.pathname)
        const normalizedDecodedPathname = decodedPathname
          .replace(/\s+/g, '')
          .replace(/appendix_m+ath/gi, canonicalSegments[0])
          .replace(/linea+r-algebra-basics/gi, canonicalSegments[1])

        if (normalizedDecodedPathname !== decodedPathname) {
          const normalizedPathname = normalizedDecodedPathname
            .split('/')
            .map((segment) => encodeURIComponent(segment))
            .join('/')
          const redirectTarget = `${normalizedPathname}${url.search}`
          res.statusCode = 302
          res.setHeader('Location', redirectTarget)
          res.end()
          return
        }

        next()
      })
    }
  }
}

function escapeHtml(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function slugifySearchHeading(value) {
  return value
    .normalize('NFKD')
    .replace(/[\u0300-\u036F]/g, '')
    .replace(/[\x00-\x1f]/g, '')
    .replace(/[\s~`!@#$%^&*()\-_+=[\]{}|\\;:"'“”‘’<>,.?/]+/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/^(\d)/, '_$1')
    .toLowerCase()
}

function stripMarkdown(value) {
  return value
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[*_~>#-]/g, '')
    .trim()
}

function renderSearchMarkdown(src) {
  const html = []
  const slugCounts = new Map()
  let inFence = false

  for (const rawLine of src.replace(/^---[\s\S]*?---\n/, '').split('\n')) {
    const line = rawLine.trim()

    if (line.startsWith('```')) {
      inFence = !inFence
      continue
    }

    if (!line || inFence || line.startsWith(':::')) {
      continue
    }

    const heading = /^(#{1,6})\s+(.+)$/.exec(line)
    if (heading) {
      const level = heading[1].length
      const title = stripMarkdown(heading[2])
      const escapedTitle = escapeHtml(title)
      const baseSlug = slugifySearchHeading(title)
      const count = slugCounts.get(baseSlug) || 0
      slugCounts.set(baseSlug, count + 1)
      const slug = count === 0 ? baseSlug : `${baseSlug}-${count}`
      html.push(
        `<h${level}>${escapedTitle}<a class="header-anchor" href="#${slug}"></a></h${level}>`
      )
      continue
    }

    html.push(`<p>${escapeHtml(stripMarkdown(line))}</p>`)
  }

  return html.join('\n')
}

function isValidMathDelimiter(state, pos) {
  const max = state.posMax
  const prevChar = pos > 0 ? state.src.charCodeAt(pos - 1) : -1
  const nextChar = pos + 1 < max ? state.src.charCodeAt(pos + 1) : -1

  return {
    canOpen: nextChar !== 0x20 && nextChar !== 0x09,
    canClose:
      prevChar !== 0x20 &&
      prevChar !== 0x09 &&
      (nextChar < 0x30 || nextChar > 0x39)
  }
}

function mathInline(state, silent) {
  if (state.src[state.pos] !== '$') return false

  // Display math inline: $$...$$ inside a paragraph/list/blockquote line.
  // Must be tried before the single-$ branch so the closing $$ isn't read as
  // two consecutive empty inline-math delimiters.
  if (state.pos + 1 < state.posMax && state.src[state.pos + 1] === '$') {
    const start = state.pos + 2
    const close = state.src.indexOf('$$', start)
    if (close !== -1 && close < state.posMax) {
      if (!silent) {
        const token = state.push('math_inline', 'math', 0)
        token.markup = '$$'
        token.content = state.src.slice(start, close)
        token.displayMode = true
      }
      state.pos = close + 2
      return true
    }
  }

  let delimiter = isValidMathDelimiter(state, state.pos)
  if (!delimiter.canOpen) {
    if (!silent) state.pending += '$'
    state.pos += 1
    return true
  }

  const start = state.pos + 1
  const max = state.posMax
  let match = start
  while ((match = state.src.indexOf('$', match)) !== -1) {
    if (match >= max) {
      match = -1
      break
    }
    let pos = match - 1
    while (state.src[pos] === '\\') pos -= 1
    if ((match - pos) % 2 === 1) break
    match += 1
  }

  if (match === -1) {
    if (!silent) state.pending += '$'
    state.pos = start
    return true
  }

  if (match - start === 0) {
    if (!silent) state.pending += '$$'
    state.pos = start + 1
    return true
  }

  delimiter = isValidMathDelimiter(state, match)
  if (!delimiter.canClose) {
    if (!silent) state.pending += '$'
    state.pos = start
    return true
  }

  if (!silent) {
    const token = state.push('math_inline', 'math', 0)
    token.markup = '$'
    token.content = state.src.slice(start, match)
  }

  state.pos = match + 1
  return true
}

function mathBlock(state, start, end, silent) {
  let pos = state.bMarks[start] + state.tShift[start]
  const max = state.eMarks[start]

  if (pos + 2 > max) return false
  if (state.src.slice(pos, pos + 2) !== '$$') return false

  pos += 2
  let firstLine = state.src.slice(pos, max)
  let lastLine = ''
  let found = false
  let next = start

  if (silent) return true
  if (firstLine.trim().slice(-2) === '$$') {
    firstLine = firstLine.trim().slice(0, -2)
    found = true
  }

  while (!found) {
    next++
    if (next >= end) break

    pos = state.bMarks[next] + state.tShift[next]
    const lineMax = state.eMarks[next]
    if (pos < lineMax && state.tShift[next] < state.blkIndent) break

    if (state.src.slice(pos, lineMax).trim().slice(-2) === '$$') {
      const lastPos = state.src.slice(0, lineMax).lastIndexOf('$$')
      lastLine = state.src.slice(pos, lastPos)
      found = true
    }
  }

  state.line = next + 1
  const token = state.push('math_block', 'math', 0)
  token.block = true
  token.content =
    (firstLine && firstLine.trim() ? `${firstLine}\n` : '') +
    state.getLines(start + 1, next, state.tShift[start], true) +
    (lastLine && lastLine.trim() ? lastLine : '')
  token.map = [start, state.line]
  token.markup = '$$'
  return true
}

function renderKatex(content, displayMode) {
  try {
    return katex.renderToString(content, {
      displayMode,
      output: 'html',
      throwOnError: true, // Enable error throwing for debugging
      strict: false,
      trust: true
    })
  } catch (error) {
    // Log detailed error information with file context
    const fs = require('fs')
    const path = require('path')
    const markdownFile = path.join(
      process.cwd(),
      'docs/chapter10_ppo/ppo-math.md'
    )

    console.error('\n' + '='.repeat(80))
    console.error('❌ KaTeX Rendering Error')
    console.error('='.repeat(80))
    console.error(
      `Mode: ${displayMode ? 'Display Math ($$...$$)' : 'Inline Math ($...$)'}`
    )
    console.error(
      `Expression: ${content.substring(0, 200)}${content.length > 200 ? '...' : ''}`
    )
    console.error(`Error: ${error.message}`)
    if (error.position !== undefined) {
      console.error(`Position in expression: ${error.position}`)
      const start = Math.max(0, error.position - 50)
      const end = Math.min(content.length, error.position + 50)
      console.error(`Context: ...${content.substring(start, end)}...`)
    }

    // Try to find this expression in the markdown file
    try {
      const mdContent = fs.readFileSync(markdownFile, 'utf8')
      const searchStr = content.substring(0, Math.min(100, content.length))
      const index = mdContent.indexOf(searchStr)
      if (index !== -1) {
        const lineNum = mdContent.substring(0, index).split('\n').length
        console.error(
          `Location: ${markdownFile}, approximately line ${lineNum}`
        )

        // Show surrounding lines
        const lines = mdContent.split('\n')
        const startLine = Math.max(0, lineNum - 3)
        const endLine = Math.min(lines.length, lineNum + 2)
        console.error('\nSurrounding context:')
        for (let i = startLine; i < endLine; i++) {
          const marker = i + 1 === lineNum ? '>>> ' : '    '
          console.error(`${marker}${i + 1}: ${lines[i].substring(0, 100)}`)
        }
      }
    } catch (e) {
      // Ignore file reading errors
    }

    console.error('='.repeat(80) + '\n')

    // Return error HTML for visibility in the page
    return `<div style="background: #fee; border: 2px solid #c00; padding: 10px; margin: 10px 0; border-radius: 4px;">
      <strong style="color: #c00;">KaTeX Rendering Error</strong>
      <p style="margin: 5px 0;"><strong>Mode:</strong> ${displayMode ? 'Display' : 'Inline'}</p>
      <p style="margin: 5px 0;"><strong>Expression:</strong> <code>${content.substring(0, 150)}${content.length > 150 ? '...' : ''}</code></p>
      <p style="margin: 5px 0;"><strong>Error:</strong> ${error.message}</p>
      <p style="margin: 5px 0; font-size: 0.9em; color: #666;">Check browser console for detailed stack trace and file location.</p>
    </div>`
  }
}

function rescueMathInInline(md) {
  // Phase 1 (core, before inline): protect $...$ from emphasis/strong rules.
  // Replace inline-math in raw token content with unique placeholders so that
  // markdown's emphasis rule never sees the underscores inside formulas.
  let mathCounter = 0
  const mathStore = new Map()

  md.core.ruler.before('inline', 'math_inline_protect', function (state) {
    for (const token of state.tokens) {
      if (token.type !== 'inline') continue
      if (!token.content.includes('$')) continue

      token.content = token.content.replace(
        /\$([^\$]+)\$/g,
        function (match, formula) {
          const key = '\x01MATH' + mathCounter++ + '\x01'
          mathStore.set(key, formula)
          return key
        }
      )
    }
  })

  // Phase 2 (core, after inline): restore placeholders as math_inline tokens.
  md.core.ruler.after('inline', 'math_inline_restore', function (state) {
    for (const token of state.tokens) {
      if (token.type !== 'inline' || !token.children) continue

      let hasPlaceholder = false
      for (const child of token.children) {
        if (child.type === 'text' && child.content.includes('\x01')) {
          hasPlaceholder = true
          break
        }
      }
      if (!hasPlaceholder) continue

      const newChildren = []
      for (const child of token.children) {
        if (child.type === 'text' && child.content.includes('\x01')) {
          const parts = child.content.split(/(\x01MATH\d+\x01)/)
          for (const part of parts) {
            if (!part) continue
            if (part.startsWith('\x01MATH') && mathStore.has(part)) {
              const t = new state.Token('math_inline', 'math', 0)
              t.content = mathStore.get(part)
              t.markup = '$'
              newChildren.push(t)
            } else if (part.trim()) {
              const t = new state.Token('text', '', 0)
              t.content = part
              newChildren.push(t)
            }
          }
        } else {
          newChildren.push(child)
        }
      }
      token.children = newChildren
    }
  })
}

function katexMarkdown(md) {
  md.inline.ruler.push('math_inline', mathInline)
  md.block.ruler.after('blockquote', 'math_block', mathBlock, {
    alt: ['paragraph', 'reference', 'blockquote', 'list']
  })
  md.renderer.rules.math_inline = (tokens, idx) =>
    renderKatex(tokens[idx].content, tokens[idx].displayMode || false)
  md.renderer.rules.math_block = (tokens, idx) =>
    `<p>${renderKatex(tokens[idx].content, true)}</p>\n`
}

function footnoteTitlePlugin(md) {
  md.renderer.rules.footnote_block_open = (tokens, idx, options, env) => {
    const previousContentToken = [...tokens]
      .slice(0, idx)
      .reverse()
      .find((token) => token.type === 'inline' && token.content?.trim())
    const previousContent = (previousContentToken?.content || '')
      .replace(/[*_`#]/g, '')
      .trim()
    const hasManualTitle = /^(参考文献|References)[:：]?$/.test(previousContent)
    const title = env.relativePath?.startsWith('en/')
      ? 'References'
      : '参考文献'
    const heading = hasManualTitle
      ? ''
      : `<div class="footnotes-title">${title}</div>\n`

    const separator = options.xhtmlOut
      ? '<hr class="footnotes-sep" />\n'
      : '<hr class="footnotes-sep">\n'

    return `${separator}<section class="footnotes">\n${heading}<ol class="footnotes-list">\n`
  }
}

function safeHeadingAttrs(md) {
  md.core.ruler.before('linkify', 'safe_heading_attrs', (state) => {
    for (let idx = 0; idx < state.tokens.length - 1; idx += 1) {
      const headingOpen = state.tokens[idx]
      const inline = state.tokens[idx + 1]

      if (headingOpen.type !== 'heading_open' || inline.type !== 'inline') {
        continue
      }

      const children = inline.children || []
      const lastText = [...children]
        .reverse()
        .find((token) => token.type === 'text')
      if (!lastText) continue

      const match = lastText.content.match(
        /\s*\{((?:[#.][A-Za-z0-9][A-Za-z0-9_.:-]*)(?:\s+[#.][A-Za-z0-9][A-Za-z0-9_.:-]*)*)\}$/
      )
      if (!match) continue

      const attrs = match[1].trim().split(/\s+/)
      const classes = []

      for (const attr of attrs) {
        if (attr.startsWith('#')) {
          headingOpen.attrSet('id', attr.slice(1))
        } else if (attr.startsWith('.')) {
          classes.push(attr.slice(1))
        }
      }

      if (classes.length) {
        headingOpen.attrJoin('class', classes.join(' '))
      }

      lastText.content = lastText.content.slice(0, match.index)
      inline.content = inline.content.replace(match[0], '')
    }
  })
}

function isExternalAsset(value) {
  return (
    /^(?:[a-z][a-z0-9+.-]*:)?\/\//i.test(value) || value.startsWith('data:')
  )
}

function resolveMarkdownAsset(relativePagePath, src) {
  if (!src || isExternalAsset(src) || src.startsWith('/')) return null

  const hashIndex = src.indexOf('#')
  const queryIndex = src.indexOf('?')
  const suffixIndexCandidates = [hashIndex, queryIndex].filter(
    (idx) => idx >= 0
  )
  const suffixIndex = suffixIndexCandidates.length
    ? Math.min(...suffixIndexCandidates)
    : -1
  const cleanSrc = suffixIndex >= 0 ? src.slice(0, suffixIndex) : src
  const suffix = suffixIndex >= 0 ? src.slice(suffixIndex) : ''
  const pageDir = path.posix.dirname(relativePagePath || '')
  const sourcePath = path.posix
    .normalize(path.posix.join(pageDir, cleanSrc))
    .replace(/^\.\//, '')

  return {
    sourcePath,
    suffix
  }
}

function optimizedImagesPlugin(md) {
  const imageRule = md.renderer.rules.image

  md.renderer.rules.image = (tokens, idx, options, env, self) => {
    const token = tokens[idx]
    const src = token.attrGet('src')
    const resolved = resolveMarkdownAsset(env.relativePath, src)
    const asset = resolved && assetManifest.assets?.[resolved.sourcePath]

    if (asset?.status === 'optimized' && asset.optimized) {
      token.attrSet('src', `${asset.optimized}${resolved.suffix}`)
      token.attrSet('data-source-src', `/src/${resolved.sourcePath}`)
    }

    if (!token.attrGet('loading')) {
      token.attrSet('loading', 'lazy')
    }

    if (!token.attrGet('decoding')) {
      token.attrSet('decoding', 'async')
    }

    return imageRule(tokens, idx, options, env, self)
  }
}

function mermaidBlockIndex(tokens, idx) {
  let index = 0

  for (let i = 0; i <= idx; i += 1) {
    const info = tokens[i].info?.trim()
    if (info === 'mermaid' || info === 'mmd') {
      index += 1
    }
  }

  return index
}

function buildMermaidManifestMap() {
  return new Map(
    (assetManifest.mermaid || [])
      .filter((block) => block.status === 'optimized' && block.optimized)
      .map((block) => [`${block.page}:${block.index}`, block])
  )
}

const mermaidManifest = buildMermaidManifestMap()

function optimizedMermaidPlugin(md) {
  const fenceRule = md.renderer.rules.fence

  md.renderer.rules.fence = (tokens, idx, options, env, self) => {
    const token = tokens[idx]
    const info = token.info.trim()

    if (info === 'mermaid' || info === 'mmd') {
      const index = mermaidBlockIndex(tokens, idx)
      const block = mermaidManifest.get(`${env.relativePath}:${index}`)

      if (block?.optimized) {
        const src = md.utils.escapeHtml(block.optimized)
        const source = md.utils.escapeHtml(block.source)
        const page = md.utils.escapeHtml(block.page)

        return `<p class="mermaid-static"><img src="${src}" alt="Mermaid diagram" data-mermaid-viewer="true" data-source-src="/src/${source}" data-source-page="/src/${page}" data-source-index="${block.index}" loading="lazy" decoding="async"></p>\n`
      }
    }

    return fenceRule(tokens, idx, options, env, self)
  }
}

const zhNav = [
  { text: '预备知识', link: '/preface/intro' },
  { text: '基础与经典 RL', link: '/chapter01_cartpole/intro' },
  { text: '深度强化学习', link: '/chapter07_dqn/from-q-to-dqn' },
  { text: '大模型对齐', link: '/chapter15_rlhf/intro' },
  { text: 'Agentic 与多模态', link: '/chapter22_agentic/intro' },
  { text: '安全与前沿', link: '/chapter30_alignment_failures/intro' },
  { text: '附录', link: '/appendix_common_pitfalls/intro' }
]

const enNav = [
  { text: 'Preface', link: '/en/preface/intro' },
  { text: 'Part I · Fundamentals', link: '/en/chapter01_cartpole/intro' },
  { text: 'Part II · Deep RL', link: '/en/chapter07_dqn/from-q-to-dqn' },
  { text: 'Part IV · LLM Alignment', link: '/en/chapter15_rlhf/intro' },
  { text: 'Part V · Agentic RL', link: '/en/chapter22_agentic/intro' },
  {
    text: 'Part VII · Safety & Frontiers',
    link: '/en/chapter32_selfplay/intro'
  },
  { text: 'Appendices', link: '/en/appendix_common_pitfalls/intro' }
]

const zhSidebar = {
  '/': [
    {
      text: '序章 · 导论',
      collapsed: false,
      items: [
        { text: '强化学习导论', link: '/preface/intro' },
        { text: '强化学习发展史', link: '/preface/brief-history/' },
        { text: '环境配置', link: '/preface/env-setup' }
      ]
    },
    {
      text: 'Part I · 基础与经典强化学习',
      collapsed: false,
      items: [
        {
          text: '1. CartPole 入门',
          link: '/chapter01_cartpole/intro',
          collapsed: true,
          items: [
            {
              text: '1.1 CartPole 原理',
              link: '/chapter01_cartpole/principles'
            },
            { text: '1.2 训练指标设计', link: '/chapter01_cartpole/metrics' },
            { text: '1.3 PPO 训练可视化', link: '/chapter01_cartpole/training' }
          ]
        },
        {
          text: '2. 多臂老虎机',
          link: '/chapter02_bandits/intro',
          collapsed: true,
          items: [
            { text: '2.1 ε-贪心策略', link: '/chapter02_bandits/intro' },
            {
              text: '2.2 UCB / Thompson 采样',
              link: '/chapter02_bandits/ucb-thompson'
            },
            {
              text: '2.3 遗憾界、PAC 与上下文老虎机',
              link: '/chapter02_bandits/theory-contextual'
            }
          ]
        },
        {
          text: '3. 马尔可夫决策过程',
          link: '/chapter03_mdp/mdp',
          collapsed: true,
          items: [
            { text: '3.1 MDP 与马尔可夫性', link: '/chapter03_mdp/mdp' },
            {
              text: '3.2 策略、价值与回报',
              link: '/chapter03_mdp/policy-value'
            },
            { text: '3.3 折扣、轨迹与 POMDP', link: '/chapter03_mdp/panorama' }
          ]
        },
        {
          text: '4. 价值函数与贝尔曼方程',
          link: '/chapter03_mdp/value-bellman',
          collapsed: true,
          items: [
            {
              text: '4.1 V/Q 函数与贝尔曼期望方程',
              link: '/chapter03_mdp/value-bellman'
            },
            {
              text: '4.2 贝尔曼最优、压缩映射与最优策略',
              link: '/chapter03_mdp/value-q'
            },
            {
              text: '4.3 价值函数数值实验',
              link: '/chapter03_mdp/value-experiment'
            }
          ]
        },
        {
          text: '5. 动态规划、蒙特卡洛与时序差分',
          link: '/chapter03_mdp/dp-mc-td',
          collapsed: true,
          items: [
            {
              text: '5.1 动态规划、蒙特卡洛、时序差分',
              link: '/chapter03_mdp/dp-mc-td'
            },
            { text: '5.2 奖励函数设计', link: '/chapter03_mdp/reward-design' }
          ]
        },
        {
          text: '6. Q-Learning 与异策略控制',
          link: '/chapter03_mdp/algorithm-taxonomy',
          collapsed: true,
          items: [
            {
              text: '6.1 on/off-policy、Q-Learning、SARSA、重要性采样与 Deadly Triad',
              link: '/chapter03_mdp/algorithm-taxonomy'
            }
          ]
        }
      ]
    },
    {
      text: 'Part II · 深度强化学习',
      collapsed: false,
      items: [
        {
          text: '7. 深度 Q 网络',
          link: '/chapter07_dqn/from-q-to-dqn',
          collapsed: true,
          items: [
            {
              text: '7.1 从 Q-Learning 到 DQN',
              link: '/chapter07_dqn/from-q-to-dqn'
            },
            { text: '7.2 DQN 改进家族', link: '/chapter07_dqn/dqn-family' },
            {
              text: '7.3 Distributional RL',
              link: '/chapter07_dqn/dqn-components'
            },
            {
              text: '7.4 LunarLander / Atari 实验',
              link: '/chapter07_dqn/lunar-lander'
            }
          ]
        },
        {
          text: '8. 策略梯度方法',
          link: '/chapter08_policy_gradient/intro',
          collapsed: true,
          items: [
            {
              text: '8.1 策略梯度定理',
              link: '/chapter08_policy_gradient/policy-gradient'
            },
            {
              text: '8.2 REINFORCE 基线',
              link: '/chapter08_policy_gradient/reinforce'
            },
            {
              text: '8.3 策略梯度改进',
              link: '/chapter08_policy_gradient/pg-improvements'
            }
          ]
        },
        {
          text: '9. Actor-Critic 架构',
          link: '/chapter09_actor_critic/actor-critic',
          collapsed: true,
          items: [
            {
              text: '9.1 优势函数',
              link: '/chapter09_actor_critic/advantage-function'
            },
            {
              text: '9.2 Actor-Critic 同步更新',
              link: '/chapter09_actor_critic/actor-critic'
            },
            { text: '9.3 实验', link: '/chapter09_actor_critic/pendulum' }
          ]
        },
        {
          text: '10. PPO 信任域方法',
          link: '/chapter10_ppo/intro',
          collapsed: true,
          items: [
            {
              text: '10.1 TRPO 信任域',
              link: '/chapter10_ppo/trust-region-clipping'
            },
            { text: '10.2 PPO-Clip 工程实现', link: '/chapter10_ppo/intro' },
            {
              text: '10.3 GAE 奖励模型',
              link: '/chapter10_ppo/gae-reward-model'
            },
            {
              text: '10.4 长程任务实验',
              link: '/chapter10_ppo/rl-long-horizon-planning'
            }
          ]
        },
        {
          text: '11. 连续控制与基于模型 RL',
          link: '/chapter11_continuous_control/intro',
          collapsed: true,
          items: [
            {
              text: '11.1 确定性策略梯度 DDPG',
              link: '/chapter11_continuous_control/intro'
            },
            {
              text: '11.2 TD3 / SAC',
              link: '/chapter11_continuous_control/td3-sac'
            },
            {
              text: '11.3 Model-Based RL 与 Dyna/PETS/MBPO',
              link: '/chapter11_continuous_control/model-based'
            },
            {
              text: '11.4 AlphaZero、MuZero 与 Dreamer V3',
              link: '/chapter11_continuous_control/search-world-models'
            }
          ]
        }
      ]
    },
    {
      text: 'Part III · 高级 RL 方法',
      collapsed: false,
      items: [
        {
          text: '12. 离线强化学习',
          link: '/chapter12_offline_rl/intro',
          collapsed: true,
          items: [
            {
              text: '12.1 离线 RL 的挑战与经典方法',
              link: '/chapter12_offline_rl/intro'
            },
            {
              text: '12.2 Decision Transformer、Trajectory Transformer 与 Diffuser',
              link: '/chapter12_offline_rl/sequence-modeling'
            },
            {
              text: '12.3 离线 RL 实验与 LLM 视角',
              link: '/chapter12_offline_rl/experiments'
            }
          ]
        },
        {
          text: '13. 模仿学习、逆向 RL 与元 RL',
          link: '/chapter13_imitation_meta_rl/bc-dagger',
          collapsed: true,
          items: [
            {
              text: '13.1 行为克隆与 DAgger',
              link: '/chapter13_imitation_meta_rl/bc-dagger'
            },
            {
              text: '13.2 逆向 RL 与 GAIL',
              link: '/chapter13_imitation_meta_rl/irl-gail'
            },
            {
              text: '13.3 元 RL 与 MAML/RL²/PEARL/In-Context RL',
              link: '/chapter13_imitation_meta_rl/meta-rl'
            }
          ]
        },
        {
          text: '14. 探索、多智能体与分层 RL',
          link: '/chapter14_exploration_marl_hierarchical/intro',
          collapsed: true,
          items: [
            {
              text: '14.1 内在动机探索与 ICM/RND/NGU/Agent57',
              link: '/chapter14_exploration_marl_hierarchical/intro'
            },
            {
              text: '14.2 多智能体 RL 与 CTDE/MADDPG/MAPPO',
              link: '/chapter14_exploration_marl_hierarchical/marl'
            },
            {
              text: '14.3 分层 RL 生成式世界模型',
              link: '/chapter14_exploration_marl_hierarchical/hierarchical'
            }
          ]
        }
      ]
    },
    {
      text: 'Part IV · 大语言模型对齐与后训练',
      collapsed: false,
      items: [
        {
          text: '15. RLHF 训练流水线',
          link: '/chapter15_rlhf/intro',
          collapsed: true,
          items: [
            {
              text: '15.1 基座模型到指令对齐',
              link: '/chapter15_rlhf/base-model-to-assistant'
            },
            {
              text: '15.2 SFT 指令微调',
              link: '/chapter15_rlhf/imitation-learning-pipeline'
            },
            {
              text: '15.3 Bradley-Terry 奖励模型',
              link: '/chapter15_rlhf/reward-function-design'
            },
            {
              text: '15.4 RL 微调流程',
              link: '/chapter15_rlhf/standard-rlhf-pipeline'
            },
            {
              text: '15.5 大规模训练工程',
              link: '/chapter15_rlhf/extended-practice'
            },
            { text: '15.6 评测方法', link: '/chapter15_rlhf/evaluation' },
            {
              text: '15.7 veRL PPO 训练 GSM8K',
              link: '/chapter15_rlhf/verl-ppo-gsm8k'
            }
          ]
        },
        {
          text: '16. 大模型 RL 工业实战',
          link: '/chapter16_llm_rl_industrial/intro',
          collapsed: true,
          items: [
            {
              text: '16.1 训练框架对比与双轨奖励',
              link: '/chapter16_llm_rl_industrial/intro'
            },
            {
              text: '16.2 现代后训练流水线范式',
              link: '/chapter16_llm_rl_industrial/industrial-post-training'
            },
            {
              text: '16.3 优化器与训练稳定性',
              link: '/chapter16_llm_rl_industrial/modern-industrial-practice'
            },
            {
              text: '16.4 分布式同步、异步与 MoE 训练',
              link: '/chapter16_llm_rl_industrial/distributed-sync'
            }
          ]
        },
        {
          text: '17. 偏好对齐与 DPO 家族',
          link: '/chapter17_dpo/intro',
          collapsed: true,
          items: [
            { text: '17.1 DPO 推导', link: '/chapter17_dpo/intro' },
            { text: '17.2 DPO 训练指标', link: '/chapter17_dpo/metrics' },
            {
              text: '17.3 DPO 原理、数学与家族选型',
              link: '/chapter17_dpo/dpo-theory-and-family'
            }
          ]
        },
        {
          text: '18. GRPO、RLVR 与 Verifier 工程',
          link: '/chapter18_grpo/grpo-practice-and-mechanism',
          collapsed: true,
          items: [
            {
              text: '18.1 GRPO 核心机制',
              link: '/chapter18_grpo/grpo-practice-and-mechanism'
            },
            {
              text: '18.2 R1-Zero 范式 / DAPO',
              link: '/chapter18_grpo/deepseek-dapo'
            },
            { text: '18.3 RLVR 可验证奖励', link: '/chapter18_grpo/rlvr' },
            { text: '18.4 GRPO 改进家族', link: '/chapter18_grpo/grpo-family' },
            {
              text: '18.5 RL Environments 与 Verifier 工程',
              link: '/chapter18_grpo/rl-environments'
            },
            {
              text: '18.6 金融 API 工具调用 GRPO 实验',
              link: '/chapter18_grpo/financial-tool-calling-grpo'
            },
            {
              text: '18.7 OPD 在线蒸馏',
              link: '/chapter18_grpo/on-policy-distillation'
            },
            {
              text: '18.8 veRL 代码生成 RL 实验',
              link: '/chapter18_grpo/verl-code-sandbox'
            }
          ]
        },
        {
          text: '19. 推理模型与 Test-Time Scaling',
          link: '/chapter19_reasoning/intro',
          collapsed: true,
          items: [
            {
              text: '19.1 推理模型的兴起',
              link: '/chapter19_reasoning/emergence-and-o1'
            },
            {
              text: '19.2 R1-Zero 纯 RL 训练',
              link: '/chapter19_reasoning/intro'
            },
            {
              text: '19.3 Test-time Compute Scaling',
              link: '/chapter19_reasoning/test-time-scaling'
            },
            {
              text: '19.4 Hybrid Thinking 思考预算',
              link: '/chapter19_reasoning/hybrid-thinking'
            },
            {
              text: '19.5 自适应思考',
              link: '/chapter19_reasoning/adaptive-thinking'
            },
            {
              text: '19.6 推理链的可读性与对齐',
              link: '/chapter19_reasoning/cot-visibility-alignment'
            }
          ]
        },
        {
          text: '20. 过程奖励模型与推理时搜索',
          link: '/chapter20_prm_search/outcome-vs-process',
          collapsed: true,
          items: [
            {
              text: '20.1 Outcome vs Process 奖励',
              link: '/chapter20_prm_search/outcome-vs-process'
            },
            {
              text: '20.2 判别式 PRM 路线',
              link: '/chapter20_prm_search/discriminative-prm'
            },
            {
              text: '20.3 生成式 PRM 路线',
              link: '/chapter20_prm_search/generative-prm'
            },
            {
              text: '20.4 形式化 PRM Verifier',
              link: '/chapter20_prm_search/formal-prm'
            },
            {
              text: '20.5 推理时搜索',
              link: '/chapter20_prm_search/inference-time-search'
            },
            {
              text: '20.6 并行协调推理总结',
              link: '/chapter20_prm_search/parallel-reasoning-and-summary'
            }
          ]
        },
        {
          text: '21. Constitutional AI 与 RLAIF',
          link: '/chapter21_cai_rlvr/intro',
          collapsed: true,
          items: [
            {
              text: '21.1 HHH 原则 Claude 实践',
              link: '/chapter21_cai_rlvr/hhh-practice'
            },
            {
              text: '21.2 RLAIF 工程化宪法扩展',
              link: '/chapter21_cai_rlvr/rlaif-engineering'
            }
          ]
        }
      ]
    },
    {
      text: 'Part V · Agentic 强化学习',
      collapsed: false,
      items: [
        {
          text: '22. 工具调用、多轮交互与多智能体 RL',
          link: '/chapter22_agentic/intro',
          collapsed: true,
          items: [
            {
              text: '22.1 Agentic RL 总览',
              link: '/chapter22_agentic/overview'
            },
            {
              text: '22.2 多轮 RL 形式化',
              link: '/chapter22_agentic/formulation'
            },
            {
              text: '22.3 轨迹信用分配',
              link: '/chapter22_agentic/credit-assignment'
            },
            {
              text: '22.4 工具调用 RL',
              link: '/chapter22_agentic/tool-use-and-trajectory'
            },
            {
              text: '22.5 Search-Augmented RL',
              link: '/chapter22_agentic/tool-use-agents'
            },
            {
              text: '22.6 Code Interpreter RL 工业实战',
              link: '/chapter22_agentic/industrial-practice'
            },
            {
              text: '22.7 多智能体协作与 Agent Swarm',
              link: '/chapter22_agentic/multi-agent-swarm'
            }
          ]
        },
        {
          text: '23. 代码智能体强化学习',
          link: '/chapter23_rl_based_swe/intro',
          collapsed: true,
          items: [
            {
              text: '23.1 SWE-RL 基础实验',
              link: '/chapter23_rl_based_swe/swe-bench-and-rlvr'
            },
            {
              text: '23.2 Code World Model 与 DeepSWE',
              link: '/chapter23_rl_based_swe/world-model-and-deep-swe'
            },
            {
              text: '23.3 Self-Play SWE-RL 总结',
              link: '/chapter23_rl_based_swe/self-play-ssr-and-summary'
            }
          ]
        },
        {
          text: '24. Deep Research 与浏览器智能体',
          link: '/chapter24_deep_research/intro',
          collapsed: true,
          items: [
            {
              text: '24.1 浏览器 RL harness 工程',
              link: '/chapter24_deep_research/browser-rl-harness'
            },
            {
              text: '24.2 评测基准与开源项目',
              link: '/chapter24_deep_research/deep-research-eval'
            }
          ]
        },
        {
          text: '25. Computer Use 与 GUI Agent',
          link: '/chapter25_computer_use/intro',
          collapsed: true,
          items: [
            {
              text: '25.1 GUI Agent 训练实践',
              link: '/chapter25_computer_use/training'
            },
            {
              text: '25.2 指令层级 Prompt Injection 防御',
              link: '/chapter25_computer_use/safety-swarm'
            }
          ]
        }
      ]
    },
    {
      text: 'Part VI · 多模态强化学习',
      collapsed: false,
      items: [
        {
          text: '26. 视觉语言模型 RL',
          link: '/chapter26_vlm/intro',
          collapsed: true,
          items: [
            {
              text: '26.1 视觉奖励挑战',
              link: '/chapter26_vlm/vlm-challenges'
            },
            {
              text: '26.2 视觉反思 RL',
              link: '/chapter26_vlm/qwen3-vl-reflection'
            },
            {
              text: '26.3 中国多模态前沿',
              link: '/chapter26_vlm/vlm-grpo-hands-on'
            },
            {
              text: '26.4 GeoQA 几何推理实验',
              link: '/chapter26_vlm/easyr1-geoqa'
            }
          ]
        },
        {
          text: '27. 音频语音 RL',
          link: '/chapter27_audio_rl/intro',
          collapsed: true,
          items: [
            {
              text: '27.1 RLVR → RLHF 音频奖励设计',
              link: '/chapter27_audio_rl/reward-design'
            },
            {
              text: '27.2 多模态音频 Agent 未来方向',
              link: '/chapter27_audio_rl/future'
            }
          ]
        },
        {
          text: '28. 具身智能与 VLA 模型',
          link: '/chapter28_vla/embodied-intelligence/'
        },
        {
          text: '29. 视觉生成 RL',
          link: '/chapter29_visual_generation/intro',
          collapsed: true,
          items: [
            {
              text: '29.1 视觉生成与 DanceGRPO',
              link: '/chapter29_visual_generation/intro'
            },
            {
              text: '29.2 多奖励视频 RLHF 与物理感知生成',
              link: '/chapter29_visual_generation/video-generation-modern'
            }
          ]
        }
      ]
    },
    {
      text: 'Part VII · 安全、评估与研究前沿',
      collapsed: false,
      items: [
        {
          text: '30. 奖励黑客与 RL 评估',
          link: '/chapter30_alignment_failures/intro',
          collapsed: true,
          items: [
            {
              text: '30.1 经典失败模式',
              link: '/chapter30_alignment_failures/classical-failures'
            },
            {
              text: '30.2 RLVR 假性收益与工业失败案例',
              link: '/chapter30_alignment_failures/modern-incidents'
            },
            {
              text: '30.3 Anthropic 失准研究',
              link: '/chapter30_alignment_failures/sleeper-and-faking'
            },
            {
              text: '30.4 防御机制总结',
              link: '/chapter30_alignment_failures/scaling-and-defenses'
            },
            {
              text: '30.5 评估原则与现代评估 Harness',
              link: '/chapter30_alignment_failures/rl-evaluation'
            }
          ]
        },
        {
          text: '31. 进化式 LLM 搜索与生成式世界模型',
          link: '/chapter31_alphaevolve/intro'
        },
        {
          text: '32. 自我博弈、规模化与未来方向',
          link: '/chapter32_selfplay/intro',
          collapsed: true,
          items: [
            {
              text: '32.1 自我博弈基础与 LLM 自我博弈',
              link: '/chapter32_selfplay/self-play-outlook/'
            },
            {
              text: '32.2 RL Scaling Laws 与 Foundation Model RL',
              link: '/chapter32_selfplay/rl-scaling-outlook'
            },
            {
              text: '32.3 In-Context RL 与未来十年',
              link: '/chapter32_selfplay/llm-multi-agent-rl/'
            }
          ]
        }
      ]
    },
    {
      text: '附录',
      items: [
        { text: 'A. 训练调试手册', link: '/appendix_common_pitfalls/intro' },
        {
          text: 'B. 强化学习工程实践',
          link: '/appendix_industrial_training/intro',
          collapsed: false,
          items: [
            {
              text: 'B.1 训练系统底座',
              link: '/appendix_industrial_training/rl-infrastructure'
            },
            {
              text: 'B.2 Agent 沙箱',
              link: '/appendix_industrial_training/agentic-rl-infra'
            },
            {
              text: 'B.3 评测基准',
              link: '/appendix_industrial_training/evaluation-badcase'
            },
            {
              text: 'B.4 训练指标词典',
              link: '/appendix_industrial_training/metrics-glossary'
            },
            {
              text: 'B.5 工业实战练习',
              link: '/appendix_industrial_training/industrial-exercises'
            }
          ]
        },
        {
          text: 'C. 核心算法实现',
          link: '/appendix_code_cheatsheet/intro',
          collapsed: false,
          items: [
            { text: 'C.1 SFT 与 KL', link: '/appendix_code_cheatsheet/sft-kl' },
            {
              text: 'C.2 PPO 与 GAE',
              link: '/appendix_code_cheatsheet/ppo-gae'
            },
            {
              text: 'C.3 DPO 家族',
              link: '/appendix_code_cheatsheet/dpo-family'
            },
            {
              text: 'C.4 GRPO 与奖励模型',
              link: '/appendix_code_cheatsheet/grpo-rlvr'
            },
            {
              text: 'C.5 Softmax 与交叉熵',
              link: '/appendix_code_cheatsheet/softmax-ce'
            },
            {
              text: 'C.6 采样方法',
              link: '/appendix_code_cheatsheet/top-k-top-p'
            },
            {
              text: 'C.7 注意力机制',
              link: '/appendix_code_cheatsheet/attention-mha'
            },
            { text: 'C.8 DAPO', link: '/appendix_code_cheatsheet/dapo' }
          ]
        },
        {
          text: 'D. 学习资源与复现项目',
          link: '/appendix_game_projects/intro'
        },
        {
          text: 'E. 数学基础',
          link: '/appendix_math/intro',
          collapsed: false,
          items: [
            {
              text: 'E.1 线性代数',
              link: '/appendix_math/linear-algebra',
              collapsed: true,
              items: [
                {
                  text: '基础对象',
                  link: '/appendix_math/linear-algebra-basics'
                },
                {
                  text: '贝尔曼矩阵',
                  link: '/appendix_math/linear-algebra-bellman'
                },
                {
                  text: '函数近似',
                  link: '/appendix_math/linear-algebra-function-approx'
                },
                {
                  text: '收敛与信任域',
                  link: '/appendix_math/linear-algebra-advanced'
                },
                {
                  text: '公式与练习',
                  link: '/appendix_math/linear-algebra-formulas-exercises'
                }
              ]
            },
            {
              text: 'E.2 概率、期望与随机估计',
              link: '/appendix_math/probability-statistics',
              collapsed: true,
              items: [
                { text: '概率基础', link: '/appendix_math/probability-basics' },
                {
                  text: '回报与价值',
                  link: '/appendix_math/probability-value'
                },
                {
                  text: '采样估计',
                  link: '/appendix_math/probability-sampling'
                },
                {
                  text: '轨迹与 GAE',
                  link: '/appendix_math/probability-trajectory-td'
                },
                {
                  text: '贝尔曼期望',
                  link: '/appendix_math/probability-bellman-advanced'
                },
                {
                  text: '公式与练习',
                  link: '/appendix_math/probability-formulas-exercises'
                }
              ]
            },
            {
              text: 'E.3 微积分与优化',
              link: '/appendix_math/calculus-optimization',
              collapsed: true,
              items: [
                { text: '导数与梯度', link: '/appendix_math/calculus-basics' },
                {
                  text: '策略梯度',
                  link: '/appendix_math/calculus-policy-gradient'
                },
                { text: 'PPO 与 Adam', link: '/appendix_math/calculus-ppo' },
                {
                  text: '推导工具',
                  link: '/appendix_math/calculus-derivations'
                },
                {
                  text: '完整公式',
                  link: '/appendix_math/calculus-advanced-formulas'
                },
                {
                  text: '公式与练习',
                  link: '/appendix_math/calculus-formulas-exercises'
                }
              ]
            },
            {
              text: 'E.4 信息论与分布距离',
              link: '/appendix_math/information-theory',
              collapsed: true,
              items: [
                { text: '熵与探索', link: '/appendix_math/information-basics' },
                {
                  text: '交叉熵与 KL',
                  link: '/appendix_math/information-cross-entropy-kl'
                },
                {
                  text: 'RLHF 与 DPO',
                  link: '/appendix_math/information-rlhf-dpo'
                },
                {
                  text: '互信息',
                  link: '/appendix_math/information-mutual-info'
                },
                {
                  text: '完整公式',
                  link: '/appendix_math/information-advanced-formulas'
                },
                {
                  text: '公式与练习',
                  link: '/appendix_math/information-formulas-exercises'
                }
              ]
            }
          ]
        },
        { text: 'F. 论文阅读路线图', link: '/appendix_paper_reading/intro' },
        { text: 'G. GPU 小时估算表', link: '/appendix_gpu_hours/intro' }
      ]
    }
  ]
}

const enSidebar = {
  '/en/': [
    {
      text: 'Preface · Introduction',
      collapsed: false,
      items: [
        { text: 'Introduction to RL', link: '/en/preface/intro' },
        { text: 'Brief History of RL', link: '/en/preface/brief-history' },
        { text: 'Environment Setup', link: '/en/preface/env-setup' }
      ]
    },
    {
      text: 'Part I · Fundamentals & Classical RL',
      collapsed: false,
      items: [
        {
          text: '1. CartPole',
          link: '/en/chapter01_cartpole/intro',
          collapsed: true,
          items: [
            {
              text: '1.1 CartPole Principles',
              link: '/en/chapter01_cartpole/principles'
            },
            {
              text: '1.2 Training Metrics',
              link: '/en/chapter01_cartpole/metrics'
            },
            { text: '1.3 PPO Training Visualization (Translation pending)' }
          ]
        },
        {
          text: '2. Multi-Armed Bandits (Translation pending)',
          collapsed: true,
          items: [
            { text: '2.1 ε-Greedy (pending)' },
            { text: '2.2 UCB / Thompson Sampling (pending)' },
            { text: '2.3 Regret, PAC & Contextual Bandits (pending)' }
          ]
        },
        {
          text: '3. Markov Decision Processes',
          link: '/en/chapter03_mdp/mdp',
          collapsed: true,
          items: [
            {
              text: '3.1 MDP & Markov Property',
              link: '/en/chapter03_mdp/mdp'
            },
            { text: '3.2 Policy, Value & Return (Translation pending)' },
            {
              text: '3.3 Discount, Trajectory & POMDP',
              link: '/en/chapter03_mdp/panorama'
            }
          ]
        },
        {
          text: '4. Value Functions & Bellman Equations',
          link: '/en/chapter03_mdp/value-bellman',
          collapsed: true,
          items: [
            {
              text: '4.1 V/Q Functions & Bellman Expectation',
              link: '/en/chapter03_mdp/value-bellman'
            },
            {
              text: '4.2 Bellman Optimality & Contraction Mapping',
              link: '/en/chapter03_mdp/value-q'
            },
            { text: '4.3 Value Function Experiments (Translation pending)' }
          ]
        },
        {
          text: '5. DP, MC & TD',
          link: '/en/chapter03_mdp/dp-mc-td',
          collapsed: true,
          items: [
            {
              text: '5.1 Dynamic Programming, Monte Carlo, Temporal Difference',
              link: '/en/chapter03_mdp/dp-mc-td'
            },
            {
              text: '5.2 Reward Function Design',
              link: '/en/chapter03_mdp/reward-design'
            }
          ]
        },
        {
          text: '6. Q-Learning & Off-Policy Control',
          link: '/en/chapter03_mdp/algorithm-taxonomy',
          collapsed: true,
          items: [
            {
              text: '6.1 On/Off-Policy, Q-Learning, SARSA, IS & Deadly Triad',
              link: '/en/chapter03_mdp/algorithm-taxonomy'
            }
          ]
        }
      ]
    },
    {
      text: 'Part II · Deep Reinforcement Learning',
      collapsed: false,
      items: [
        {
          text: '7. Deep Q-Networks',
          link: '/en/chapter07_dqn/from-q-to-dqn',
          collapsed: true,
          items: [
            {
              text: '7.1 From Q-Learning to DQN',
              link: '/en/chapter07_dqn/from-q-to-dqn'
            },
            {
              text: '7.2 DQN Improvement Family',
              link: '/en/chapter07_dqn/dqn-family'
            },
            {
              text: '7.3 Distributional RL',
              link: '/en/chapter07_dqn/dqn-components'
            },
            {
              text: '7.4 LunarLander / Atari Experiments',
              link: '/en/chapter07_dqn/lunar-lander'
            }
          ]
        },
        {
          text: '8. Policy Gradient Methods',
          link: '/en/chapter08_policy_gradient/intro',
          collapsed: true,
          items: [
            {
              text: '8.1 Policy Gradient Theorem',
              link: '/en/chapter08_policy_gradient/policy-gradient'
            },
            {
              text: '8.2 REINFORCE with Baseline',
              link: '/en/chapter08_policy_gradient/reinforce'
            },
            {
              text: '8.3 Policy Gradient Improvements',
              link: '/en/chapter08_policy_gradient/pg-improvements'
            }
          ]
        },
        {
          text: '9. Actor-Critic Architecture',
          link: '/en/chapter09_actor_critic/actor-critic',
          collapsed: true,
          items: [
            {
              text: '9.1 Advantage Function',
              link: '/en/chapter09_actor_critic/advantage-function'
            },
            {
              text: '9.2 Actor-Critic Synchronous Updates',
              link: '/en/chapter09_actor_critic/actor-critic'
            },
            {
              text: '9.3 Pendulum Experiments',
              link: '/en/chapter09_actor_critic/pendulum'
            }
          ]
        },
        {
          text: '10. PPO & Trust-Region Methods',
          link: '/en/chapter10_ppo/intro',
          collapsed: true,
          items: [
            {
              text: '10.1 TRPO Trust Region',
              link: '/en/chapter10_ppo/trust-region-clipping'
            },
            {
              text: '10.2 PPO-Clip Implementation',
              link: '/en/chapter10_ppo/intro'
            },
            {
              text: '10.3 GAE & Reward Model',
              link: '/en/chapter10_ppo/gae-reward-model'
            },
            {
              text: '10.4 Long-Horizon Task Experiments',
              link: '/en/chapter10_ppo/rl-long-horizon-planning'
            }
          ]
        },
        {
          text: '11. Continuous Control & Model-Based RL (Translation pending)',
          collapsed: true,
          items: [
            { text: '11.1 DDPG (pending)' },
            { text: '11.2 TD3 / SAC (pending)' },
            { text: '11.3 Model-Based RL: Dyna / PETS / MBPO (pending)' },
            { text: '11.4 AlphaZero, MuZero & Dreamer V3 (pending)' }
          ]
        }
      ]
    },
    {
      text: 'Part III · Advanced RL Methods',
      collapsed: false,
      items: [
        {
          text: '12. Offline Reinforcement Learning (Translation pending)',
          collapsed: true,
          items: [
            {
              text: '12.1 Offline RL Challenges & Classical Methods (pending)'
            },
            {
              text: '12.2 Decision Transformer, Trajectory Transformer & Diffuser (pending)'
            },
            { text: '12.3 Offline RL Experiments & LLM Perspective (pending)' }
          ]
        },
        {
          text: '13. Imitation, Inverse RL & Meta-RL (Translation pending)',
          collapsed: true,
          items: [
            { text: '13.1 Behavioral Cloning & DAgger (pending)' },
            { text: '13.2 Inverse RL & GAIL (pending)' },
            {
              text: '13.3 Meta-RL: MAML / RL² / PEARL / In-Context RL (pending)'
            }
          ]
        },
        {
          text: '14. Exploration, MARL & Hierarchical RL (Translation pending)',
          collapsed: true,
          items: [
            {
              text: '14.1 Intrinsic Motivation: ICM / RND / NGU / Agent57 (pending)'
            },
            { text: '14.2 Multi-Agent RL: CTDE / MADDPG / MAPPO (pending)' },
            { text: '14.3 Hierarchical RL & Generative World Models (pending)' }
          ]
        }
      ]
    },
    {
      text: 'Part IV · LLM Alignment & Post-Training',
      collapsed: false,
      items: [
        {
          text: '15. RLHF Pipeline',
          link: '/en/chapter15_rlhf/intro',
          collapsed: true,
          items: [
            {
              text: '15.1 Base Model to Instruction Alignment',
              link: '/en/chapter15_rlhf/base-model-to-assistant'
            },
            {
              text: '15.2 SFT Instruction Tuning',
              link: '/en/chapter15_rlhf/imitation-learning-pipeline'
            },
            {
              text: '15.3 Bradley-Terry Reward Model',
              link: '/en/chapter15_rlhf/reward-function-design'
            },
            {
              text: '15.4 RL Fine-Tuning Pipeline',
              link: '/en/chapter15_rlhf/standard-rlhf-pipeline'
            },
            {
              text: '15.5 Large-Scale Training Engineering',
              link: '/en/chapter15_rlhf/extended-practice'
            },
            {
              text: '15.6 Evaluation Methods',
              link: '/en/chapter15_rlhf/evaluation'
            },
            {
              text: '15.7 veRL PPO on GSM8K',
              link: '/en/chapter15_rlhf/verl-ppo-gsm8k'
            }
          ]
        },
        {
          text: '16. Industrial LLM RL Practice',
          link: '/en/chapter16_llm_rl_industrial/industrial-post-training',
          collapsed: true,
          items: [
            {
              text: '16.1 Training Frameworks & Dual-Track Rewards (Translation pending)'
            },
            {
              text: '16.2 Modern Post-Training Pipeline Paradigms',
              link: '/en/chapter16_llm_rl_industrial/industrial-post-training'
            },
            {
              text: '16.3 Optimizers & Training Stability (Translation pending)'
            },
            {
              text: '16.4 Distributed Sync/Async & MoE Training (Translation pending)'
            }
          ]
        },
        {
          text: '17. Preference Alignment & DPO Family',
          link: '/en/chapter17_dpo/intro',
          collapsed: true,
          items: [
            { text: '17.1 DPO Derivation', link: '/en/chapter17_dpo/intro' },
            {
              text: '17.2 DPO Training Metrics',
              link: '/en/chapter17_dpo/metrics'
            },
            {
              text: '17.3 DPO Theory, Math & Family Selection',
              link: '/en/chapter17_dpo/dpo-theory-and-family'
            }
          ]
        },
        {
          text: '18. GRPO, RLVR & Verifier Engineering',
          link: '/en/chapter18_grpo/grpo-practice-and-mechanism',
          collapsed: true,
          items: [
            {
              text: '18.1 GRPO Core Mechanism',
              link: '/en/chapter18_grpo/grpo-practice-and-mechanism'
            },
            {
              text: '18.2 R1-Zero Paradigm / DAPO',
              link: '/en/chapter18_grpo/deepseek-dapo'
            },
            {
              text: '18.3 RLVR: Verifiable Rewards',
              link: '/en/chapter18_grpo/rlvr'
            },
            { text: '18.4 GRPO Improvement Family (Translation pending)' },
            {
              text: '18.5 RL Environments & Verifier Engineering (Translation pending)'
            },
            {
              text: '18.6 Financial API Tool-Calling GRPO Experiment',
              link: '/en/chapter18_grpo/financial-tool-calling-grpo'
            },
            {
              text: '18.7 On-Policy Distillation',
              link: '/en/chapter18_grpo/on-policy-distillation'
            },
            {
              text: '18.8 veRL Code Generation RL Experiment (Translation pending)'
            }
          ]
        },
        {
          text: '19. Reasoning Models & Test-Time Scaling (Translation pending)',
          collapsed: true,
          items: [
            { text: '19.1 Emergence of Reasoning Models (pending)' },
            { text: '19.2 R1-Zero Pure RL Training (pending)' },
            { text: '19.3 Test-time Compute Scaling (pending)' },
            { text: '19.4 Hybrid Thinking & Thinking Budget (pending)' },
            { text: '19.5 Adaptive Thinking (pending)' },
            { text: '19.6 CoT Readability & Alignment (pending)' }
          ]
        },
        {
          text: '20. Process Reward Models & Inference-Time Search (Translation pending)',
          collapsed: true,
          items: [
            { text: '20.1 Outcome vs Process Rewards (pending)' },
            { text: '20.2 Discriminative PRM (pending)' },
            { text: '20.3 Generative PRM (pending)' },
            { text: '20.4 Formal PRM Verifier (pending)' },
            { text: '20.5 Inference-Time Search (pending)' },
            { text: '20.6 Parallel Reasoning Coordination (pending)' }
          ]
        },
        {
          text: '21. Constitutional AI & RLAIF (Translation pending)',
          collapsed: true,
          items: [
            { text: '21.1 HHH Principles & Claude Practice (pending)' },
            { text: '21.2 RLAIF Engineering Constitution Extension (pending)' }
          ]
        }
      ]
    },
    {
      text: 'Part V · Agentic Reinforcement Learning',
      collapsed: false,
      items: [
        {
          text: '22. Tool Use, Multi-Turn & Multi-Agent RL',
          link: '/en/chapter22_agentic/intro',
          collapsed: true,
          items: [
            {
              text: '22.1 Agentic RL Overview (Translation pending)'
            },
            {
              text: '22.2 Multi-Turn RL Formulation (Translation pending)'
            },
            {
              text: '22.3 Trajectory Credit Assignment (Translation pending)'
            },
            {
              text: '22.4 Tool-Use RL',
              link: '/en/chapter22_agentic/tool-use-and-trajectory'
            },
            {
              text: '22.5 Search-Augmented RL',
              link: '/en/chapter22_agentic/tool-use-agents'
            },
            {
              text: '22.6 Code Interpreter RL Industrial Practice',
              link: '/en/chapter22_agentic/industrial-practice'
            },
            {
              text: '22.7 Multi-Agent Collaboration & Agent Swarm (Translation pending)'
            }
          ]
        },
        {
          text: '23. RL for Code Agents (Translation pending)',
          collapsed: true,
          items: [
            { text: '23.1 SWE-RL Basics (pending)' },
            { text: '23.2 Code World Model & DeepSWE (pending)' },
            { text: '23.3 Self-Play SWE-RL Summary (pending)' }
          ]
        },
        {
          text: '24. Deep Research & Browser Agents (Translation pending)',
          collapsed: true,
          items: [
            { text: '24.1 Browser RL Harness Engineering (pending)' },
            {
              text: '24.2 Evaluation Benchmarks & Open-Source Projects (pending)'
            }
          ]
        },
        {
          text: '25. Computer Use & GUI Agents (Translation pending)',
          collapsed: true,
          items: [
            { text: '25.1 GUI Agent Training Practice (pending)' },
            {
              text: '25.2 Instruction Hierarchy & Prompt Injection Defense (pending)'
            }
          ]
        }
      ]
    },
    {
      text: 'Part VI · Multimodal Reinforcement Learning',
      collapsed: false,
      items: [
        {
          text: '26. Vision-Language Model RL',
          link: '/en/chapter26_vlm/intro',
          collapsed: true,
          items: [
            {
              text: '26.1 Visual Reward Challenges',
              link: '/en/chapter26_vlm/vlm-challenges'
            },
            {
              text: '26.2 Visual Reflection RL',
              link: '/en/chapter26_vlm/vlm-grpo-hands-on'
            },
            { text: '26.3 Multimodal Frontiers (Translation pending)' },
            {
              text: '26.4 GeoQA Geometric Reasoning Experiment',
              link: '/en/chapter26_vlm/easyr1-geoqa'
            }
          ]
        },
        {
          text: '27. Audio & Speech RL (Translation pending)',
          collapsed: true,
          items: [
            { text: '27.1 RLVR → RLHF Audio Reward Design (pending)' },
            { text: '27.2 Multimodal Audio Agent Future Directions (pending)' }
          ]
        },
        {
          text: '28. Embodied Intelligence & VLA Models (Translation pending)',
          collapsed: true,
          items: [{ text: '28.1 Embodied Intelligence Overview (pending)' }]
        },
        {
          text: '29. Visual Generation RL (Translation pending)',
          collapsed: true,
          items: [
            { text: '29.1 Visual Generation & DanceGRPO (pending)' },
            {
              text: '29.2 Multi-Reward Video RLHF & Physics-Aware Generation (pending)'
            }
          ]
        }
      ]
    },
    {
      text: 'Part VII · Safety, Evaluation & Research Frontiers',
      collapsed: false,
      items: [
        {
          text: '30. Reward Hacking & RL Evaluation (Translation pending)',
          collapsed: true,
          items: [
            { text: '30.1 Classical Failure Modes (pending)' },
            {
              text: '30.2 RLVR Fake Gains & Industrial Failure Cases (pending)'
            },
            { text: '30.3 Anthropic Misalignment Research (pending)' },
            { text: '30.4 Defense Mechanisms Summary (pending)' },
            { text: '30.5 Evaluation Principles & Modern Harnesses (pending)' }
          ]
        },
        {
          text: '31. Evolutionary LLM Search & Generative World Models (Translation pending)',
          collapsed: true,
          items: [{ text: '31.1 AlphaEvolve (pending)' }]
        },
        {
          text: '32. Self-Play, Scaling & Future Directions',
          link: '/en/chapter32_selfplay/intro',
          collapsed: true,
          items: [
            {
              text: '32.1 Self-Play Basics & LLM Self-Play',
              link: '/en/chapter32_selfplay/self-play-outlook/'
            },
            {
              text: '32.2 RL Scaling Laws & Foundation Model RL',
              link: '/en/chapter32_selfplay/rl-scaling-outlook'
            },
            {
              text: '32.3 In-Context RL & the Next Decade',
              link: '/en/chapter32_selfplay/llm-multi-agent-rl/'
            }
          ]
        }
      ]
    },
    {
      text: 'Appendices',
      collapsed: false,
      items: [
        {
          text: 'A. Training Debugging Guide',
          link: '/en/appendix_common_pitfalls/intro'
        },
        {
          text: 'B. RL Engineering Practice',
          link: '/en/appendix_industrial_training/intro',
          collapsed: false,
          items: [
            {
              text: 'B.1 Training Infrastructure',
              link: '/en/appendix_industrial_training/rl-infrastructure'
            },
            {
              text: 'B.2 Agent Sandbox',
              link: '/en/appendix_industrial_training/agentic-rl-infra'
            },
            {
              text: 'B.3 Evaluation Benchmarks',
              link: '/en/appendix_industrial_training/evaluation-badcase'
            },
            {
              text: 'B.4 Metrics Glossary',
              link: '/en/appendix_industrial_training/metrics-glossary'
            },
            {
              text: 'B.5 Industrial Exercises',
              link: '/en/appendix_industrial_training/industrial-exercises'
            }
          ]
        },
        {
          text: 'C. Core Algorithm Implementations',
          link: '/en/appendix_code_cheatsheet/intro',
          collapsed: false,
          items: [
            {
              text: 'C.1 SFT and KL',
              link: '/en/appendix_code_cheatsheet/sft-kl'
            },
            {
              text: 'C.2 PPO and GAE',
              link: '/en/appendix_code_cheatsheet/ppo-gae'
            },
            {
              text: 'C.3 DPO Family',
              link: '/en/appendix_code_cheatsheet/dpo-family'
            },
            {
              text: 'C.4 GRPO and Reward Models',
              link: '/en/appendix_code_cheatsheet/grpo-rlvr'
            },
            {
              text: 'C.5 Softmax & Cross-Entropy',
              link: '/en/appendix_code_cheatsheet/softmax-ce'
            },
            {
              text: 'C.6 Sampling Methods',
              link: '/en/appendix_code_cheatsheet/top-k-top-p'
            },
            {
              text: 'C.7 Attention Mechanism',
              link: '/en/appendix_code_cheatsheet/attention-mha'
            },
            { text: 'C.8 DAPO', link: '/en/appendix_code_cheatsheet/dapo' }
          ]
        },
        {
          text: 'D. Learning Resources & Reproduction Projects',
          link: '/en/appendix_game_projects/intro'
        },
        {
          text: 'E. Math Foundations',
          link: '/en/appendix_math/intro',
          collapsed: false,
          items: [
            {
              text: 'E.1 Linear Algebra',
              link: '/en/appendix_math/linear-algebra',
              collapsed: true,
              items: [
                {
                  text: 'Basic Objects',
                  link: '/en/appendix_math/linear-algebra-basics'
                },
                {
                  text: 'Bellman Matrix',
                  link: '/en/appendix_math/linear-algebra-bellman'
                },
                {
                  text: 'Function Approximation',
                  link: '/en/appendix_math/linear-algebra-function-approx'
                },
                {
                  text: 'Convergence & Trust Regions',
                  link: '/en/appendix_math/linear-algebra-advanced'
                },
                {
                  text: 'Formulas & Exercises',
                  link: '/en/appendix_math/linear-algebra-formulas-exercises'
                }
              ]
            },
            {
              text: 'E.2 Probability & Estimation',
              link: '/en/appendix_math/probability-statistics',
              collapsed: true,
              items: [
                {
                  text: 'Probability Basics',
                  link: '/en/appendix_math/probability-basics'
                },
                {
                  text: 'Returns and Value',
                  link: '/en/appendix_math/probability-value'
                },
                {
                  text: 'Sampling & Estimation',
                  link: '/en/appendix_math/probability-sampling'
                },
                {
                  text: 'Trajectories and GAE',
                  link: '/en/appendix_math/probability-trajectory-td'
                },
                {
                  text: 'Bellman Expectations',
                  link: '/en/appendix_math/probability-bellman-advanced'
                },
                {
                  text: 'Formulas & Exercises',
                  link: '/en/appendix_math/probability-formulas-exercises'
                }
              ]
            },
            {
              text: 'E.3 Calculus & Optimization',
              link: '/en/appendix_math/calculus-optimization',
              collapsed: true,
              items: [
                {
                  text: 'Derivatives & Gradients',
                  link: '/en/appendix_math/calculus-basics'
                },
                {
                  text: 'Policy Gradient',
                  link: '/en/appendix_math/calculus-policy-gradient'
                },
                {
                  text: 'PPO and Adam',
                  link: '/en/appendix_math/calculus-ppo'
                },
                {
                  text: 'Derivation Tools',
                  link: '/en/appendix_math/calculus-derivations'
                },
                {
                  text: 'Complete Formulas',
                  link: '/en/appendix_math/calculus-advanced-formulas'
                },
                {
                  text: 'Formulas & Exercises',
                  link: '/en/appendix_math/calculus-formulas-exercises'
                }
              ]
            },
            {
              text: 'E.4 Information Theory',
              link: '/en/appendix_math/information-theory',
              collapsed: true,
              items: [
                {
                  text: 'Entropy & Exploration',
                  link: '/en/appendix_math/information-basics'
                },
                {
                  text: 'Cross-Entropy & KL',
                  link: '/en/appendix_math/information-cross-entropy-kl'
                },
                {
                  text: 'RLHF and DPO',
                  link: '/en/appendix_math/information-rlhf-dpo'
                },
                {
                  text: 'Mutual Information',
                  link: '/en/appendix_math/information-mutual-info'
                },
                {
                  text: 'Complete Formulas',
                  link: '/en/appendix_math/information-advanced-formulas'
                },
                {
                  text: 'Formulas & Exercises',
                  link: '/en/appendix_math/information-formulas-exercises'
                }
              ]
            }
          ]
        },
        { text: 'F. Paper Reading Roadmap (Translation pending)' },
        { text: 'G. GPU Hours Estimation Table (Translation pending)' }
      ]
    }
  ]
}

const logger = createLogger()
const originalWarn = logger.warn
logger.warn = (msg, options) => {
  if (msg.includes('Failed to resolve "/@siteData"')) return
  originalWarn(msg, options)
}

export default defineConfig({
  lang: 'zh-CN',
  title: 'Hands-on Modern RL',
  description:
    '现代强化学习实战指南：涵盖经典控制、LLM 后训练、RLVR 与多模态智能体',
  base,
  cleanUrls: true,
  lastUpdated: true,
  srcExclude: ['**/_archive_v5.1/**', '**/_archive/**'],
  markdown: {
    image: {
      lazyLoading: true
    },
    attrs: {
      disable: true
    },
    config: (md) => {
      safeHeadingAttrs(md)
      optimizedImagesPlugin(md)
      md.use(markdownItFootnote)
      footnoteTitlePlugin(md)
      katexMarkdown(md)
      MermaidMarkdown(md)
      optimizedMermaidPlugin(md)
      // Custom "output" container for displaying code running results
      md.use(markdownItContainer, 'output', {
        render: function (tokens, idx) {
          if (tokens[idx].nesting === 1) {
            const title = tokens[idx].info.trim().slice(6).trim() || '运行结果'
            return `<div class="custom-block output"><p class="custom-block-title">${title}</p>\n`
          }
          return '</div>\n'
        }
      })
    }
  },
  vite: {
    customLogger: logger,
    server: {
      watch: {
        ignored: ['**/.vitepress/dist/**']
      }
    },
    plugins: [
      mermaidConfigPlugin(),
      normalizeBrokenDocPathPlugin()
    ],
    optimizeDeps: {
      include: [
        '@braintree/sanitize-url',
        'cytoscape',
        'cytoscape-cose-bilkent',
        'dayjs',
        'debug'
      ]
    },
    resolve: {
      alias: {
        'dayjs/plugin/advancedFormat.js': 'dayjs/esm/plugin/advancedFormat',
        'dayjs/plugin/customParseFormat.js':
          'dayjs/esm/plugin/customParseFormat',
        'dayjs/plugin/isoWeek.js': 'dayjs/esm/plugin/isoWeek',
        'cytoscape/dist/cytoscape.umd.js': 'cytoscape/dist/cytoscape.esm.js'
      }
    }
  },
  ignoreDeadLinks: true,
  head: [
    ['link', { rel: 'icon', href: `${base}favicon.svg` }],
    ['meta', { name: 'theme-color', content: '#3f51b5' }],
    [
      'meta',
      { name: 'viewport', content: 'width=device-width, initial-scale=1.0' }
    ],
    ['meta', { name: 'author', content: 'WalkingLabs' }],
    ['meta', { name: 'robots', content: 'index,follow' }],
    ['meta', { property: 'og:title', content: 'Hands-on Modern RL' }],
    [
      'meta',
      {
        property: 'og:description',
        content:
          '现代强化学习实战指南：涵盖经典控制、LLM 后训练、RLVR 与多模态智能体'
      }
    ],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:url', content: siteUrl }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }]
  ],
  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/',
      title: 'Hands-on Modern RL',
      description: '现代强化学习实战——从代码到原理',
      themeConfig: {
        nav: zhNav,
        sidebar: zhSidebar,
        editLink: {
          pattern: editLinkPattern,
          text: '在 GitHub 上编辑此页'
        },
        footer: {
          message: '现代强化学习实战课程',
          copyright: 'Copyright © WalkingLabs'
        },
        outline: {
          level: [2, 3],
          label: '大纲'
        },
        lastUpdated: {
          text: '最后更新'
        },
        docFooter: {
          prev: '上一页',
          next: '下一页'
        },
        darkModeSwitchLabel: '外观',
        lightModeSwitchTitle: '切换到浅色模式',
        darkModeSwitchTitle: '切换到深色模式',
        sidebarMenuLabel: '菜单',
        returnToTopLabel: '返回顶部',
        langMenuLabel: '切换语言',
        skipToContentLabel: '跳转到正文',
        notFound: {
          title: '页面未找到',
          quote: '这个地址不存在，试试从中文首页重新进入。',
          link: '/',
          linkText: '返回中文首页',
          linkLabel: '返回中文首页'
        }
      }
    },
    en: {
      label: 'English',
      lang: 'en-US',
      link: '/en/',
      title: 'Hands-on Modern RL',
      description:
        'Modern Reinforcement Learning in Practice — From Code to Theory',
      themeConfig: {
        nav: enNav,
        sidebar: enSidebar,
        editLink: {
          pattern: editLinkPattern,
          text: 'Edit this page on GitHub'
        },
        footer: {
          message: '现代强化学习实战课程',
          copyright: 'Copyright © WalkingLabs'
        },
        outline: {
          level: [2, 3],
          label: '大纲'
        },
        lastUpdated: {
          text: 'Last updated'
        },
        docFooter: {
          prev: 'Previous page',
          next: 'Next page'
        },
        darkModeSwitchLabel: 'Appearance',
        lightModeSwitchTitle: 'Switch to light theme',
        darkModeSwitchTitle: 'Switch to dark theme',
        sidebarMenuLabel: 'Menu',
        returnToTopLabel: 'Return to top',
        langMenuLabel: 'Change language',
        skipToContentLabel: 'Skip to content',
        notFound: {
          title: 'Page not found',
          quote:
            'This page is missing. Try jumping back in from the English home page.',
          link: '/en/',
          linkText: 'Take me to English home',
          linkLabel: 'Go to English home'
        }
      }
    }
  },
  themeConfig: {
    logo: '/readme/logo-symbol.svg',
    siteTitle: 'Hands on Modern RL',
    nav: zhNav,
    sidebar: zhSidebar,
    socialLinks: [
      { icon: 'github', link: `https://github.com/${owner}/${repo}` }
    ],
    search: enableLocalSearch
      ? {
          provider: 'local',
          options: {
            _render: renderSearchMarkdown
          }
        }
      : undefined,
    editLink: {
      pattern: editLinkPattern,
      text: 'Edit this page on GitHub'
    },
    footer: {
      message: 'Built for reusable bilingual course delivery',
      copyright: 'Copyright © WalkingLabs'
    },
    outline: {
      level: [2, 3],
      label: '大纲'
    }
  }
})
