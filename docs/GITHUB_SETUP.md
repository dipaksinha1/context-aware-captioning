# GitHub repository and local development

The repository is https://github.com/dipaksinha1/context-aware-captioning.
Clone it to continue development. The creation instructions below apply only to a separate copy;
do not run repository creation again for this existing repository.

```bash
git clone https://github.com/dipaksinha1/context-aware-captioning.git
cd context-aware-captioning
```

Suggested name: `context-aware-captioning`. Use **private** initially for coursework.
The source is already arranged as a normal Python Git repository. No GitHub account name is assumed.

## With GitHub CLI

Install Git and [GitHub CLI](https://cli.github.com/), then run from this project folder:

```bash
gh auth login
git status
```

If you downloaded the ZIP, it excludes `.git`, so initialize it:

```bash
git init -b main
git add .
git commit -m "Build speaker-aware captioning prototype"
```

If Git requests your author identity, configure it using your own GitHub name and email.
Do not use example credentials or someone else's identity.

Create the repository and upload:

```bash
gh repo create context-aware-captioning --private --source=. --remote=origin --push --description "Context-aware captioning for multi-speaker media (COSC-5926)"
```

Only run repository creation once. If that name is already taken in your account, choose a different
name before running it. If the repository already exists, use its **actual copied URL**:

```bash
git remote add origin YOUR_ACTUAL_REPOSITORY_URL
git push -u origin main
```

Replace the placeholder with the URL GitHub shows. Do not push until `git status` shows only intended
project source. `.gitignore` excludes tokens, uploaded media, model weights, and outputs; the one bundled
synthetic demonstration WAV is explicitly allowed.

## Without GitHub CLI

1. Open <https://github.com/new> while signed into your account.
2. Choose the repository name above and **Private**. Do not pre-create a README, license, or gitignore
   because the local project includes its own source and README.
3. Initialize/commit locally using the commands above, if needed.
4. Follow GitHub's “push an existing repository” instructions using its actual repository URL.
5. In the Actions tab, verify the Python checks workflow completes. No Hugging Face secrets are needed
   by these tests because real models are not downloaded in CI.

## Next milestones

- Run and record the real-model smoke test.
- Annotate consented reference recordings and evaluate word/speaker/timestamp errors.
- Conduct the HCI review tasks in `EVALUATION.md`.
- Choose a source-code license before public release; model licenses remain separate.
