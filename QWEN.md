# Agent Rules

- Work only inside this workspace unless the user explicitly names another path.
- Do not read files under Documents, Desktop, Downloads, Library, .ssh, .codex, or .qwen.
- Do not use network tools or web fetch tools during the initial safety tests.
- Do not delete files or directories.
- Do not overwrite original media files.
- Show the intended file changes before editing existing files.
- Ask before running commands that change existing files.
- Keep durable generated results under ./outputs when media or batch outputs are involved.
- Keep temporary inputs, extracted frames, and scratch files under ./work.
- Keep each user-facing reproduction script beside its durable output, and keep shared helpers under scripts/.
