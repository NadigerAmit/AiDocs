# Aider + Ollama + Qwen2.5-Coder: Complete Local AI Coding Assistant Guide

## Introduction

Modern AI coding assistants such as GitHub Copilot, Claude Code, Cursor, and Gemini Code Assist are powerful tools. However, many developers prefer a fully local solution for:

- Source code privacy
- Offline development
- Zero API costs
- No subscription fees
- Full control over the AI stack

One of the most effective local coding setups currently available is:

```text
Aider
+
Ollama
+
Qwen2.5-Coder
+
VS Code
```

This combination provides an AI coding assistant capable of understanding repositories, modifying code, generating documentation, and helping with development tasks, all without sending source code to external services.

---

# Architecture Overview

```text
┌─────────────────┐
│     VS Code     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      Aider      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Ollama      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Qwen2.5-Coder   │
│      14B        │
└─────────────────┘
         │
         ▼
   Git Repository
```

---

# What Is Aider?

Aider is an open-source AI pair-programmer that works directly with your Git repository.

Capabilities:

- Read project files
- Edit project files
- Create new files
- Refactor code
- Generate unit tests
- Create documentation
- Update multiple files simultaneously
- Work with Git

Unlike traditional chatbots, Aider is specifically optimized for software development workflows.

---

# What Is Ollama?

Ollama is a local large language model runtime.

It allows developers to run AI models directly on their machine without using cloud APIs.

Popular models include:

- Qwen2.5-Coder
- DeepSeek-Coder
- Gemma
- Mistral
- Llama

Benefits:

- Offline access
- Privacy
- No token charges
- No internet dependency

---

# Why Choose Qwen2.5-Coder?

Qwen2.5-Coder is one of the strongest open-source coding models currently available.

Excellent support for:

- Java
- Kotlin
- Android
- Spring Boot
- Rust
- C++
- Python
- Docker
- Kubernetes
- SQL

---

# Recommended Hardware

## Minimum

```text
16 GB RAM
6 CPU Cores
SSD Storage
```

## Recommended

```text
32 GB RAM
Intel Core Ultra / Ryzen 7
NVMe SSD
```

## Ideal

```text
64 GB RAM
Dedicated GPU
NVMe SSD
```

---

# Selecting The Correct Model

## Qwen2.5-Coder 32B

Pros:

- Better reasoning
- Better architecture analysis
- Handles larger codebases

Cons:

- Very high RAM usage
- Generates more heat
- Slower responses
- High CPU load

Typical RAM Usage:

```text
20-30 GB
```

---

## Qwen2.5-Coder 14B

Pros:

- Faster
- Lower RAM consumption
- Better user experience
- Works comfortably on 32 GB systems

Cons:

- Slightly weaker reasoning

Typical RAM Usage:

```text
8-12 GB
```

Recommended for most developers.

---

# Installing Ollama

Ubuntu:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify:

```bash
ollama --version
```

Check service:

```bash
systemctl --user status ollama
```

Start if necessary:

```bash
systemctl --user start ollama
```

---

# Download Qwen2.5-Coder

Install:

```bash
ollama pull qwen2.5-coder:14b
```

Verify:

```bash
ollama list
```

Expected:

```text
qwen2.5-coder:14b
```

Test:

```bash
ollama run qwen2.5-coder:14b
```

Example:

```text
Write a Java Hello World program.
```

Exit:

```text
/bye
```

---

# Installing Python

Ubuntu:

```bash
sudo apt update

sudo apt install python3-pip python3-venv -y
```

Verify:

```bash
python3 --version
```

---

# Installing Aider

Create virtual environment:

```bash
mkdir ~/aider

cd ~/aider

python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

Install Aider:

```bash
pip install -U aider-chat
```

Verify:

```bash
aider --version
```

---

# Initial Configuration

Create configuration directory:

```bash
mkdir -p ~/.config/aider
```

Create file:

```bash
nano ~/.config/aider/.aider.conf.yml
```

Recommended configuration:

```yaml
model: ollama/qwen2.5-coder:14b

yes-always: false

auto-commits: false

dark-mode: true
```

---

# Configuration Explanation

## model

```yaml
model: ollama/qwen2.5-coder:14b
```

Specifies which model Aider should use.

---

## yes-always

```yaml
yes-always: false
```

Recommended.

Aider asks before changing files.

Example:

```text
May I modify these files?
```

This protects your repository.

Do NOT use initially:

```yaml
yes-always: true
```

---

## auto-commits

```yaml
auto-commits: false
```

Recommended for production repositories.

Why?

Because:

```text
AI might make mistakes.
```

You should always review changes before committing.

Unsafe configuration:

```yaml
auto-commits: true
```

Potential outcome:

```text
AI edits files
AI commits files
Build breaks
```

Therefore beginners should keep:

```yaml
auto-commits: false
```

---

## dark-mode

```yaml
dark-mode: true
```

Purely visual preference.

Better terminal readability.

---

# Starting Aider

Navigate into repository:

```bash
cd ~/Projects/DICE
```

Launch:

```bash
aider --model ollama/qwen2.5-coder:14b
```

Expected startup:

```text
Repo: DICE
Model: qwen2.5-coder:14b
```

---

# Recommended Git Workflow

Before any significant change:

```bash
git checkout -b aider-feature
```

Examples:

```bash
git checkout -b docker-support
```

```bash
git checkout -b artwork-analysis
```

This allows safe experimentation.

---

# Essential Aider Commands

## Add Files

Add specific files:

```text
/add path/to/file
```

Example:

```text
/add server/src/main/java
```

---

## Show Added Files

```text
/files
```

---

## Show Differences

```text
/diff
```

Displays all generated changes.

---

## Undo Last Change

```text
/undo
```

---

## Clear Context

```text
/clear
```

Very important for local models.

Reduces memory usage.

---

## Exit

```text
/exit
```

---

# Best Practices

## Keep Scope Small

Good:

```text
Analyze artwork module only.
```

Bad:

```text
Analyze my entire repository.
```

---

## Be Explicit

Bad Prompt:

```text
Where is project configuration?
```

Good Prompt:

```text
Search only the server module.

Find:

1. application.yaml
2. application.yml
3. application.properties

Report:
- file path
- purpose
- which one is primary

Do not modify files.
```

---

## Review Changes

Always run:

```text
/diff
```

before accepting modifications.

---

## Build Frequently

Java:

```bash
mvn clean install
```

or

```bash
gradle build
```

Verify generated code actually works.

---

# Example Prompts

## Analyze Configuration

```text
Analyze configuration structure.

Show:

1. Main configuration file
2. Environment overrides
3. YAML hierarchy
4. Application startup flow

Do not modify files.
```

---

## Analyze Spring Boot Application

```text
Analyze server startup.

Show:

1. Main application class
2. Configuration loading
3. Bean registration
4. REST endpoint registration

Do not modify files.
```

---

## Dockerize Application

```text
Create:

1. Dockerfile
2. docker-compose.yml
3. README deployment instructions

Requirements:

- Java 21
- Multi-stage build
- Production ready
```

---

## Generate Unit Tests

```text
Generate JUnit tests.

Include:

1. Success cases
2. Failure cases
3. Edge cases
4. Mock dependencies
```

---

## Analyze Architecture

```text
Analyze artwork processing module.

Show:

1. REST endpoints
2. Services
3. Repositories
4. Database entities
5. Cache usage

Do not modify files.
```

---

# Monitoring Performance

Check model memory usage:

```bash
ollama ps
```

Check system memory:

```bash
free -h
```

Check running processes:

```bash
htop
```

Monitor CPU temperatures:

```bash
watch sensors
```

Install sensors if needed:

```bash
sudo apt install lm-sensors
```

---

# Advantages

## Cost

```text
No subscriptions
No API fees
```

## Privacy

```text
Source code stays local
```

## Offline

```text
Works without internet
```

## Git Aware

```text
Repository focused
Version controlled
```

## Multi-file Editing

Can update entire features spanning many files.

---

# Disadvantages

## Requires Good Hardware

Large models consume significant RAM.

---

## Slower Than Cloud Models

Claude, GPT-5, and Gemini are generally faster.

---

## More Manual Setup

Requires installation and configuration.

---

## Higher Laptop Temperatures

Expect:

- Higher CPU usage
- Fan activity
- Increased power consumption

---

# Recommended Setup For 32 GB RAM Machines

Hardware:

```text
32 GB RAM
Intel Core Ultra
Ryzen 7
NVMe SSD
```

Recommended Stack:

```text
VS Code
+
Git
+
Ollama
+
Qwen2.5-Coder 14B
+
Aider
```

Recommended Model:

```text
qwen2.5-coder:14b
```

Recommended Aider Settings:

```yaml
model: ollama/qwen2.5-coder:14b

yes-always: false

auto-commits: false

dark-mode: true
```

This provides one of the best balances currently available between:

- Cost
- Privacy
- Performance
- Code quality
- Local execution
- Resource consumption

for professional software development on modern laptops.

---

# Operational Notes: Reliability, Privacy, And Model Drift

This guide describes one local-tool configuration, not a permanent recommendation. Model tags, context limits, quantization, Aider/Ollama configuration keys, licensing, and hardware requirements change. Before installing, verify the current Aider and Ollama documentation, the exact model tag and license, and the available RAM/VRAM on the target machine. Treat the hardware table and model-quality statements above as estimates for the stated setup, not portable benchmarks.

## What “Local” Does And Does Not Mean

Local inference can avoid sending prompts to a hosted model endpoint, but it does not automatically guarantee privacy or security. Review the configured provider/base URL, model-pull source, logs and shell history, editor extensions, telemetry settings, repository permissions, file access, and any tools the assistant can invoke. Do not include credentials, patient data, or proprietary files unless your policy explicitly permits it. Keep model artifacts and private inputs out of Git.

## Safe Change Loop

1. Start from a clean or understood Git worktree and state the requested scope.
2. Add only the files needed for context; exclude secrets, generated data, and unrelated large files.
3. Ask for a small change with expected behavior and tests, and request clarification when requirements are ambiguous.
4. Inspect the complete diff, including new files and configuration changes.
5. Run the relevant tests, formatters, and security checks yourself; a model's claim that a command passed is not evidence that it ran.
6. Review dependencies, shell commands, network access, destructive operations, and generated credentials before execution.
7. Commit only after human review, with a clear message and a recoverable diff.

Avoid automatic command approval and automatic commits for an unfamiliar repository. Keep a separate backup or rely on reviewed Git history; never assume a coding model will preserve every behavior not named in a prompt.

## Prompt Template For Engineering Tasks

```text
Goal: [observable behavior to add or fix]
Scope: [files/components allowed to change]
Constraints: [APIs, compatibility, security, performance]
Evidence: [failing test, expected output, or reproduction]
Acceptance checks: [commands/behaviors to verify]
Before editing: summarize the controlling code path and the smallest test.
After editing: report changed files, tests actually run, and remaining uncertainty.
```

This template improves task clarity, but the generated patch still needs the same review and verification as a human-written change.

For the broader learning sequence, see the [AI Engineering Study Roadmap](STUDY_ROADMAP.md).
