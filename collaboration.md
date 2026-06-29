# Git Collaboration Guide

## 🚨 Important Rules

* **Never commit directly to ****`main`****.**
* **Always work on your own branch.**
* **Pull the latest changes before starting work.**
* **Push only your own branch.**

---

# Step 1: Clone the repository (Only once)

```bash
git clone <repository-url>
cd <repository-name>
```

---

# Step 2: Create a separate branch for each feature , or just one single thats yours

Replace `<feature-name>` with what you are updating or adding.

```bash
git checkout -b <feature-name>
```

Example:

```bash
git checkout -b ECM-pipeline
```

Push your branch once:

```bash
git push -u origin <feature-name>
```

---

# Step 3: Before you start coding (Every time)

Switch to the main branch and get the latest code.

```bash
git checkout main
git pull origin main
```

Now switch back to your branch.

```bash
git checkout <feature-name>
```

Merge the latest changes from `main`.

```bash
git merge main
```

Now start coding.

---

# Step 4: Save your work

```bash
git add .
git commit -m "Short description of what you changed"
```

Examples:

```bash
git commit -m "Added login page"
git commit -m "Fixed navbar bug"
git commit -m "Created API for authentication"
```

---

# Step 5: Push your changes

```bash
git push origin <feature-name>
```

---

# Step 6: Open a Pull Request

1. Go to GitHub.
2. Open your branch.
3. Click **Compare & Pull Request**.
4. Create the Pull Request.
5. Wait till we discuss and then we can merge it

**Do NOT merge your own Pull Request without discussing.**

---

# Daily Workflow

```text
Pull latest main
        ↓
Switch to your branch
        ↓
Merge main into your branch
        ↓
Code
        ↓
Commit
        ↓
Push
        ↓
Create Pull Request
```

---

# Things NOT to do ❌

* Don't push directly to `main`.
* Don't work on someone else's branch.
* Don't force push (`git push --force`).
* Don't delete someone else's branch.
* Don't commit unfinished or broken code.

---

# If you get stuck

Don't randomly try Git commands.

Ask in the team group before doing anything that might affect the repository.
