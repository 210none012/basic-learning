# 实验一：Git 基础操作

## 一、实验目的

1. 理解 Git 版本控制的基本概念
2. 掌握 Git 的常用命令
3. 学会使用远程仓库进行代码管理

---

## 二、实验环境

- 操作系统：Windows
- 工具：Git (命令行)
- 远程仓库：GitHub (https://github.com/210none012/basic-learning.git)

---

## 三、实验原理

Git 是一个分布式版本控制系统，核心分为三个区域：

| 区域    | 说明                                   |
| ----- | ------------------------------------ |
| 工作区   | 存放实际文件的目录，即我们编辑代码的地方                 |
| 暂存区   | 临时存放修改的区域，通过 git add 将文件从工作区提交到暂存区   |
| 本地仓库  | 保存所有提交历史的地方，通过 git commit 将暂存区内容永久记录 |

文件在 Git 中有三种状态：**已修改** → **已暂存** → **已提交**

远程交互流程：

工作区  ---add--->  暂存区  ---commit--->  本地仓库  ---push--->  远程仓库
工作区  <--merge/checkout---  暂存区  <---reset---  本地仓库  <---fetch---  远程仓库

---

## 四、实验步骤与过程

### 步骤 1：配置 Git 用户信息

首先需要配置用户名和邮箱，这些信息会记录在每次提交中。

### 设置用户名
git config --global user.name "名字"

### 设置邮箱
git config --global user.email "邮箱"

### github操作

1. 在home界面点击左边栏New创建新存储库。
2. 输入存储库名字，并点击创建。

### 步骤 2：添加远程仓库

将本地仓库与 GitHub 远程仓库关联。

git remote add origin https://github.com/210none012/basic-learning.git

执行结果：成功添加名为 origin 的远程仓库。

验证：
git remote -v

输出：

origin  https://github.com/210none012/basic-learning.git (fetch)
origin  https://github.com/210none012/basic-learning.git (push)

### 步骤 3：从远程拉取代码

git fetch origin

执行结果：

From https://github.com/210none012/basic-learning
 * [new branch]      main       -> origin/main

git fetch 将远程仓库的最新信息下载到本地，但不会自动合并到当前工作分支。

### 步骤 4：切换并跟踪远程分支

git checkout main

执行结果：

branch 'main' set up to track 'origin/main'.
Already on 'main'

此时本地 main 分支已与远程 origin/main 建立跟踪关系，后续 git pull 和 git push 可以省略远程名和分支名。

### 步骤 5：创建实验目录和文件

mkdir day1
 创建 day1/node.md 文件
 
## VS code操作

1. 在github上创建仓库，并复制项目地址。
2. 点击左边栏Source Control(git图标)。
3. 点击Clone Repository，并在上方粘贴项目地址。
4. 选择项目克隆到哪一个文件夹。
5. 点击commit，publish（pull）即可提交修改。
6. 点击Sync Changes可同步远程仓库。
---

## 五、常用 Git 命令及知识汇总

### 基础概念

1. repository：仓库，在Github等平台的为远程仓库，在本地的为本地仓库
    - 可分为Work Directory(工作区)、Local Repository(本地仓库)、Remote Repository(远程仓库)、Staging/Index(暂存区)三个区域。
    - git clone将远程仓库保存到本地，同时创建出本地存储库和工作目录。
    - git add+git commit将工作区改动提交到本地仓库。
    - git add将文件保存到暂存区，git commit将暂存区所有改动提交到本地仓库
    - git push将本地仓库改动推送到远程仓库。
    - git pull将远程仓库最新改动更新并合并到本地
    - git pull = git fetch + git merge
2. branch：在不同分支上的修改不会互相影响，开发结束后，即可将分支合并回主干（merge）
3. remote：

### 配置类
| 命令 | 说明 |
|------|------|
| git config --global user.name "name" | 设置用户名 |
| git config --global user.email "email" | 设置邮箱 |

### 仓库类
| 命令 | 说明 |
|------|------|
| git init | 初始化本地仓库 |
| git clone <url> | 克隆远程仓库到本地 |

### 日常操作
| 命令 | 说明 |
|------|------|
| git status | 查看工作区和暂存区状态 |
| git add <file> | 将文件添加到暂存区 |
| git add . | 添加所有修改到暂存区 |
| git commit -m "message" | 提交暂存区内容到本地仓库 |
| git log | 查看提交历史 |
| git diff | 查看未暂存的修改内容 |

### 分支类
| 命令 | 说明 |
|------|------|
| git branch | 查看本地分支列表 |
| git branch <name> | 创建新分支 |
| git checkout <name> | 切换到指定分支 |
| git checkout -b <name> | 创建并切换到新分支 |
| git merge <name> | 将指定分支合并到当前分支 |

### 远程类
| 命令 | 说明 |
|------|------|
| git remote add <name> <url> | 添加远程仓库 |
| git remote -v | 查看远程仓库列表 |
| git fetch <remote> | 从远程获取最新信息（不合并） |
| git pull <remote> <branch> | 拉取远程分支并合并到本地 |
| git push <remote> <branch> | 推送本地分支到远程 |

### 其他

1. 初始化后，会在本地文件夹中生成.git文件夹（隐藏），存储与git有关数据。
2. .gitignore文件中声明了文件夹中那些文件不受git管理（如密钥，依赖包等文件），文件为文件名，目录为目录名/。
3. git reset --hard CommitID可用于把仓库强制回退到某历史状态（用于单人使用的分支还未提交到远程仓库）。
4. discard用于放弃还没commit的文件修改。
5. revert生成反向commit抵消某次commit（用于多人协作分支，相对安全）
6. 

---

## 六、实验结果

1. 成功将本地仓库与远程 GitHub 仓库关联
2. 成功从远程仓库拉取代码到本地
3. 本地分支 main 已跟踪远程分支 origin/main
4. 掌握了 Git 工作区 → 暂存区 → 本地仓库 → 远程仓库的完整工作流

---

## 七、实验总结

通过本次实验，我学习到了以下要点：

- **Git 的核心思想**：分布式版本控制，每个开发者都有完整的仓库副本
- **标准工作流**：add → commit → push，修改文件后先添加到暂存区，再提交到本地仓库，最后推送到远程
- **fetch vs pull**：fetch 只下载不合并，pull = fetch + merge
- **分支的作用**：可以在不同分支上并行开发，互不影响，开发完成后再合并
- **提交信息**：每次 commit 都应该写清楚改了什么，方便日后回溯和协作
