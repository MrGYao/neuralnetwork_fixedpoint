import utils from './utils.js'
import path from 'path'
import fs from 'fs-extra'

const COMPONENT_MAP_LOCAL = {
  agent: { src: 'agent', dest: 'agent' },
  commands: { src: 'commands', dest: 'commands' },
  skills: { src: 'skills', dest: 'skill' },
}

const COMPONENT_MAP_GLOBAL = {
  core: { src: '', dest: '' },
}

async function installLocal(projectRoot, components, options = {}) {
  const { force = false } = options
  const corePath = utils.getCorePath(projectRoot)
  const localPath = utils.getLocalInstallPath(projectRoot)

  utils.log.info(`本项目安装模式`)
  utils.log.info(`源路径: ${corePath}`)
  utils.log.info(`目标路径: ${localPath}`)

  const results = []

  for (const comp of components) {
    const mapping = COMPONENT_MAP_LOCAL[comp]
    if (!mapping) {
      utils.log.error(`未知组件: ${comp}`)
      continue
    }

    utils.log.info(`安装组件: ${comp}`)

    let copied = false
    let destPath

    // 特殊处理 agent 组件：合并 agent/ 和 agents/
    if (comp === 'agent') {
      const agentPath = path.join(corePath, 'agent')
      const agentsPath = path.join(corePath, 'agents')
      destPath = path.join(localPath, 'agent')

      const agentExists = await fs.pathExists(agentPath)
      const agentsExists = await fs.pathExists(agentsPath)

      if (agentExists && agentsExists) {
        // 两个目录都存在：先复制 agent/，再合并 agents/
        utils.log.info(`检测到 agent/ 和 agents/ 存在，合并复制`)
        await utils.copyDir(agentPath, destPath, { force })
        // 第二次用 fs.copy 直接合并，不删除目标目录
        const items = await fs.readdir(agentsPath)
        for (const item of items) {
          const srcItem = path.join(agentsPath, item)
          const destItem = path.join(destPath, item)
          await fs.copy(srcItem, destItem)
        }
        copied = true
      } else if (agentExists) {
        // 只存在 agent/
        await utils.copyDir(agentPath, destPath, { force })
        copied = true
      } else if (agentsExists) {
        // 只存在 agents/
        await utils.copyDir(agentsPath, destPath, { force })
        copied = true
      } else {
        utils.log.warn(`未找到 agent/ 或 agents/ 目录，跳过`)
      }
    } else {
      // 其他组件：正常复制
      const srcPath = path.join(corePath, mapping.src)
      destPath = path.join(localPath, mapping.dest)

      const exclude = comp === 'skills' ? [] : []
      copied = await utils.copyDir(srcPath, destPath, { force, exclude })
    }

    results.push({ component: comp, copied, path: destPath })
  }

  await utils.recordInstall(projectRoot, components, 'local')

  return results
}

async function installGlobal(projectRoot, components, options = {}) {
  const { force = false } = options
  const corePath = utils.getCorePath(projectRoot)
  const globalPath = utils.getGlobalInstallPath()

  utils.log.info(`全局安装模式`)
  utils.log.info(`源路径: ${corePath}`)
  utils.log.info(`目标路径: ${globalPath}`)

  await utils.copyDir(corePath, globalPath, { force, exclude: ['install'] })

  await utils.recordInstall(projectRoot, ['all'], 'global')

  return [{ component: 'all', copied: true, path: globalPath }]
}

export default {
  installLocal,
  installGlobal,
  COMPONENT_MAP_LOCAL,
  COMPONENT_MAP_GLOBAL,
}
