# GitHub 协作快捷指令

本文件将用户的自然语言请求映射到 Agent 的 Git 命令及 GitHub 操作流程。它是操作指南，不是可直接执行的脚本，也不会注册终端别名。

## 1. 项目与工具

- 仓库：<https://github.com/gududegenbai3/stereo-vision-perception>
- 远程名：`origin`；默认分支：`main`。
- `git`：本地版本管理、分支、拉取、推送与本地合并。
- GitHub 插件：查询仓库、PR、Issue 和 CI，以及在授权范围内创建、更新和合并 PR。
- `gh`：可选的 GitHub CLI，提供与插件部分重叠的命令行功能；不是执行 `git pull/push` 的必要条件。

2026-10-20 已核验：Git 可用，GitHub 插件已安装并启用，且能访问本仓库；当前连接账号拥有仓库管理、拉取和推送权限。`git ls-remote origin` 已成功读取远程 `main`。本机尚未找到 `gh`。插件授权不等于本机 Git 已登录，终端推送能力需由实际操作验证；本次未执行推送测试。

插件属于使用者的 Codex 环境，不是 Python 项目依赖；组员克隆仓库后，需要分别连接自己的 GitHub 账号及获得仓库权限。

## 2. 操作前的上下文检查

在仓库根目录执行：

```sh
git status --short
git branch --show-current
git remote -v
git diff --stat
git diff --cached --stat
```

Agent 根据检查结果确定当前分支、未提交内容和操作范围。下文的 `<branch>`、`<source_branch>`、`<target_branch>`、`<pr_number>`、`<head_sha>` 等均为占位符，执行时替换成实际值。

用户明确提出拉取、推送、提交或合并时，Agent 在该请求范围内执行，不将请求仅作为命令咨询。账号登录、系统权限或仓库规则阻止执行时，报告具体原因。创建本文件本身不触发提交、推送或合并。

## 3. 自然语言 → CLI / 插件操作

### “帮我从 GitHub 云端同步到本地” / “拉取最新代码”

默认更新当前分支对应的远程分支；若用户明确指定主分支，则在适合切换分支的工作区状态下切换到 `main`。

```sh
git fetch origin
git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}'
git pull --ff-only
```

`--ff-only` 只接受快进更新，避免一次普通同步隐式生成合并提交。当前分支没有 upstream 时，先确定对应远程分支；不要直接将 `origin/main` 拉入任意功能分支。存在未提交内容时先检查影响，保护本地修改；会覆盖文件或需要保存方式决策时，说明具体文件及处理选择。历史分叉时转入合并流程，不通过强制重置覆盖本地工作。

### “帮我看看云端有哪些更新”

```sh
git fetch origin
git log --oneline HEAD..'@{upstream}'
git diff --stat HEAD...'@{upstream}'
```

仅查看更新，不修改当前分支文件。这里同样以当前分支已配置 upstream 为前提。

### “帮我提交本次修改”

检查完整 diff，确定本次任务的文件后，使用 commit 技能生成提交说明，按明确文件路径暂存并提交：

```sh
git diff
git diff --cached
git add -- <file_1> <file_2>
git diff --cached
git commit -m '<commit_message>'
```

已有暂存内容可能来自其他协作者或任务，先核对范围。提交只写入本地历史；用户只要求提交时，不额外推送。

### “帮我推送到 GitHub” / “上传本地提交”

```sh
git branch --show-current
git push -u origin <branch>
```

推送当前分支已存在的提交。未提交文件不会被 push 上传；如果只有未提交改动，先说明状态。用户要求“提交并推送本次修改”时，连续执行提交与推送流程。推送遭拒时先获取远程状态并判断分叉原因，不自动使用强推。

### “帮我新建一个分支开发这个功能”

以当前项目约定使用 `codex/` 前缀，例如 `codex/stereo-capture`；用户指定名称时使用指定名称。

在工作区干净、从最新主分支开始时：

```sh
git switch main
git pull --ff-only origin main
git switch -c <branch>
```

若用户希望将当前未提交工作移到新分支，则从当前状态执行 `git switch -c <branch>`，不先切换或拉取主分支。

### “帮我创建 PR”

先验证修改并推送功能分支，再通过 GitHub 插件创建以 `main` 为目标的 PR；标题和说明包含具体改动及验证结果。成功后将 PR 链接附加到当前 Codex 聊天。

已安装并登录 `gh` 时，对应 CLI 为：

```sh
gh pr create --base main --head <branch> --title '<title>' --body-file <body_file>
```

`<body_file>` 是 Agent 预先写好的说明文件。发布 PR 属于 GitHub 写操作，依据用户的创建请求执行。

### “帮我看看 PR / CI 状态”

优先使用 GitHub 插件读取指定 PR、检查结果和评审意见；需要明确 PR 时，根据用户提供的编号或当前分支匹配，避免选择不相关的 PR。

可选 CLI：

```sh
gh pr view <pr_number>
gh pr checks <pr_number>
```

### “帮我合并这个 PR”

先核对 PR 的源分支、目标分支、最新提交、检查及评审状态，再使用 GitHub 插件合并，并传入已核对的 head SHA。默认使用 merge commit；仓库规则或用户指定其他方式时按相应方式执行。遵守 GitHub 的保护规则与所需检查。

可选 CLI：

```sh
gh pr merge <pr_number> --merge --match-head-commit <head_sha>
```

云端合并后，如果本地当前就是目标分支且工作区干净，则拉取更新；否则报告云端已合并及本地尚未更新的状态。用户未要求时不额外删除分支。

### “把 A 分支合并到 B 分支”

这是本地分支合并，区别于合并 GitHub PR。先 fetch 并确认 A、B 分别指本地还是远程分支，再在干净工作区执行：

```sh
git switch <target_branch>
git merge <source_branch>
```

遇到冲突时检查文件含义，解决能够明确判断的部分并验证；需要选择功能行为时向用户说明具体冲突。目标是受保护主分支时优先走 PR。合并不会自动推送，除非用户同时要求推送。

## 4. GitHub 插件怎么用

插件已在当前环境启用，可直接在聊天中指定本项目和需求：

- “用 GitHub 插件列出这个项目未合并的 PR。”
- “查看这个项目 PR #12 的修改、检查结果和评审意见。”
- “把当前功能分支推送并创建 PR，目标是 main。”
- “检查 PR #12，如果满足仓库合并条件就合并。”

Agent 按工具能力组合执行：本地文件和 Git 历史由终端处理，云端 PR、Issue 和检查由插件处理。云端 PR 合并后，本地文件仍需通过 Git 同步。

组员在各自 Codex 环境的插件管理中查找并连接 GitHub，完成账号授权并开放本仓库访问；组织仓库可能还需要管理员批准。无需把插件加入 Python 依赖文件，也不把账号授权信息写入仓库。

## 5. 可选的 GitHub CLI 初始化

本机使用 Homebrew 时，安装命令为：

```sh
brew install gh
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status
```

安装与登录是独立的环境配置步骤，本文件不代表已经执行。网页登录由账号持有人完成。即使插件已经连接，`gh` 仍需要自己的登录状态；未安装 `gh` 时，优先使用已有插件完成支持的云端操作。

## 6. 参考

- [Git pull：快进与分支同步](https://git-scm.com/docs/git-pull)
- [GitHub CLI：PR 合并及 head SHA 校验](https://cli.github.com/manual/gh_pr_merge)
- [GitHub CLI：登录](https://cli.github.com/manual/gh_auth_login)
- [OpenAI 插件概览](https://developers.openai.com/plugins)
