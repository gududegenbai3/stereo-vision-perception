# Codex + GitHub 个人贡献流程

本流程沿用项目 README 的分支和 PR 约定，供协作者在自己的本地副本中使用。

## 首次配置

在项目目录确认提交身份，邮箱使用自己 GitHub 账户已关联的邮箱：

```powershell
git config --local user.name "你的提交姓名"
git config --local user.email "你的提交邮箱"
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status
```

提交姓名和邮箱决定提交作者；GitHub 登录决定推送权限，两者需分别配置。
如 Git 无法联网而浏览器可用，检查浏览器的本机代理，并按实际地址配置项目级 `http.proxy`。代理端口变化时同步更新；停止使用代理后可执行 `git config --local --unset http.proxy`。

## 每个任务

工作区干净时，从最新的团队主分支创建个人任务分支：

```powershell
git status --short --branch
git fetch origin
git switch -c 你的用户名/任务名 origin/main
```

如果已有未提交改动，先识别并保存这些改动，再创建下一个任务分支。

给 Codex 的任务应包含目标、修改范围和验收标准，例如：

> 请先读取 README 和有关文档，完成这个任务。只修改本任务需要的文件，保留其他人的改动。运行与改动相关的验证，检查差异，再提交到当前个人分支。PR 中说明修改内容、验证结果和未验证项。

完成后先审查差异，再明确选择本次文件提交：

```powershell
git diff
git diff --check
git add docs/本次修改的文件.md
git diff --cached
git commit -m "docs: 说明本次改动"
git push -u origin HEAD
gh pr create --base main --title "说明本次改动" --body-file PR说明文件的路径
```

上述文件路径是示例，执行时替换为真实文件。PR 说明文件放在仓库外或未跟踪的临时目录中，避免随代码提交。

## 验证和审查

- 文档改动检查链接、事实、格式与差异；代码加入后按项目实际工具运行构建和必要测试。
- 摄像头采集、标定和深度结果需要实机验证，PR 明确记录未完成的硬件验证。
- PR 写明目标、改动、验证、已知未知项和风险，请团队成员审查后按团队规则合并。
- 审查期间继续提交到同一任务分支，PR 会随推送更新。
- 遇到冲突时先检查双方意图，再处理冲突并重新验证。
- 每次提交只包含本任务文件；不要覆盖他人改动、强推共享分支，或将密钥和本机私有配置提交到仓库。
- 大型采集数据和实验产物的位置遵循项目目录约定。

合并后开始下一项任务时重新从 `origin/main` 创建分支。
