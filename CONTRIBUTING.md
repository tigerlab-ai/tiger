# Contribution Guide
Thanks for helping out! We're excited for your issues and pull requests! 

In addition to the mentioned roadmap, we also maintain a backlog at https://github.com/tigerlab-ai/tiger/issues.

# Setup
1. Fork the TigerLab repository
2. Clone the fork to your local machine
3. Set the upstream remote to the original TigerLab repository
```bash
git clone https://github.com/tigerlab-ai/tiger.git
cd tiger
git remote add upstream https://github.com/tigerlab-ai/tiger.git
```
4. Install `pre-commit` and set up the `pre-commit` hooks for the repo:
```bash
pip install pre-commit
pre-commit install
```

# Making Changes
Before making changes, ensure that you're working with the most recent version of the code:

```bash
git checkout main
git pull upstream main
```

Create a new branch for your changes:

```bash
git checkout -b <branch-name>
```

# Coding Standards
Please adhere to the following:

1. Follow existing coding style and conventions.
2. Write clear, readable, and maintainable code.
3. Include detailed comments when necessary.
4. Test your changes thoroughly before submitting your pull request.

# Committing Your Changes
1. Stage your changes: git add .
2. Commit your changes: git commit -m "Your detailed commit message"
3. Push your changes to your fork: git push origin <branch-name>

# Submitting a Pull Request
1. Navigate to the original TigerLab repository
2. Click 'New pull request'
3. Choose your fork and the branch containing your changes
4. Give your pull request a title and detailed description
5. Submit the pull request

# Issue Tracking
If you're adding a new feature or fixing a bug, make sure to add or update an issue in the issue tracker. If there's not an existing issue, create a new one. This helps us keep track of what needs to be worked on.

# Code of Conduct
By participating in this project, you're expected to uphold our Code of Conduct.

# Where to Get Help
If you run into problems or need help, check out our [discord](https://discord.gg/GnwH2STv).

Thank you for considering contributing to TigerLab! Your time and expertise is greatly appreciated by the community.
