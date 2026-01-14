# Local Development Guide

Quick reference for running your Jekyll homepage + Hugo blog locally.

## Start Full Site (Homepage + Blog)

```bash
cd /mnt/d/projects/hsiaotsan.github.io
sudo ./build-all.sh serve
```

Then visit:
- **Homepage:** http://localhost:4000
- **Blog:** http://localhost:4000/blog/

This builds Hugo first, then starts Jekyll server which serves both.

---

## Start Blog Only (Faster for blog editing)

```bash
cd /mnt/d/projects/hsiaotsan.github.io
./blog-dev.sh
```

Then visit:
- **Blog:** http://localhost:1313/blog/

This has **live reload** - changes appear instantly!

---

## Quick Comparison

| Command | Use When | URL | Features |
|---------|----------|-----|----------|
| `sudo ./build-all.sh serve` | Testing full site, navigation between homepage and blog | localhost:4000 | Full Jekyll + Hugo |
| `./blog-dev.sh` | Writing blog posts | localhost:1313/blog/ | Live reload, shows drafts |

---

## Creating New Blog Posts

### 1. Create a new post

```bash
cd /mnt/d/projects/hsiaotsan.github.io/blog
hugo new posts/your-post-name.md
```

### 2. Edit the post

Open `blog/content/posts/your-post-name.md` and edit:

```markdown
---
title: "Your Post Title"
date: 2025-12-31T...
draft: false              # Change to false when ready to publish
tags: ["ai", "research"]
categories: ["Tech"]
description: "Brief description for SEO"
summary: "Post summary"
---

Your content here...
```

### 3. Preview locally

```bash
./blog-dev.sh
```

Visit http://localhost:1313/blog/ to see your changes in real-time.

### 4. Publish

```bash
git add blog/content/posts/your-post-name.md
git commit -m "Add new blog post: Your Post Title"
git push origin main
```

GitHub Actions will automatically build and deploy (takes 1-2 minutes).

---

## Stop Server

Press `Ctrl+C` in the terminal

Or if running in background:
```bash
sudo pkill -f jekyll    # Stop Jekyll
sudo pkill -f hugo      # Stop Hugo
```

---

## Adding Images

1. Place images in: `blog/static/images/`
2. Reference in posts: `![Alt text](/images/filename.jpg)`

Example:
```markdown
![My Research Diagram](/images/research-diagram.png)
```

---

## Tips

- **Production URLs:** The production site uses `https://hsiaotsan.github.io/blog/`
- **Local development:** The Hugo dev server (`./blog-dev.sh`) is best for writing since it auto-reloads
- **Full preview:** Use `./build-all.sh serve` before pushing to verify navigation and layout
- **Draft posts:** Posts with `draft: true` are visible in `blog-dev.sh` but hidden in production

---

## Troubleshooting

### Port already in use
```bash
sudo lsof -ti:4000 | xargs sudo kill -9
sudo lsof -ti:1313 | xargs sudo kill -9
```

### Changes not showing
- For blog changes: Run `./blog-dev.sh` instead of `./build-all.sh serve`
- For homepage changes: Jekyll auto-regenerates on save
- **WSL Note:** The `blog-dev.sh` script uses `--poll 1s` to work around WSL file system limitations. Changes may take 1-2 seconds to appear after saving

### Blog 404 on production
- Ensure `baseURL` in `blog/config.yml` is set to `https://hsiaotsan.github.io/blog/`
- Check GitHub Actions completed successfully

---

## Project Structure

```
hsiaotsan.github.io/
├── _config.yml              # Jekyll config
├── _data/navigation.yml     # Site navigation
├── blog/                    # Hugo blog source
│   ├── config.yml          # Hugo config
│   ├── content/posts/      # Blog posts go here
│   ├── static/images/      # Blog images
│   └── themes/PaperMod/    # Blog theme
├── build-all.sh            # Full site build script
├── blog-dev.sh             # Hugo dev server script
└── _site/                  # Generated site (git-ignored)
    └── blog/               # Hugo output
```

---

## Deployment

Every push to `main` triggers GitHub Actions to:
1. Install Hugo Extended
2. Build Hugo blog → `_site/blog/`
3. Install Jekyll dependencies
4. Build Jekyll (preserves Hugo blog)
5. Deploy to GitHub Pages

Monitor builds at: https://github.com/HsiaoTsan/hsiaotsan.github.io/actions
