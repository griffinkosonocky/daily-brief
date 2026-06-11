# The Daily Brief

The Daily Brief is a static personal morning site. A scheduled GitHub Action runs a Python script, asks the Anthropic API for the generated sections, writes plain HTML into `site/`, commits the updated files, and deploys that folder to GitHub Pages.

The site is designed for phone reading first: dark mode, a narrow reading width, a compact market card at the top, hidden sports trivia answer, a long historical read with a Leaflet map, and a fixed scrolling news ticker.

## How it Works

- `generate.py` builds the daily edition.
- `ticker.py` updates `site/ticker.json` from RSS feeds without using an LLM.
- `topics.json` stores the historical topic queue.
- `trivia_history.json` stores the last 90 trivia questions to reduce repeats.
- `state.json` stores the edition number and the last usable section content.
- `site/index.html` is today's page.
- `site/archive/YYYY-MM-DD.html` is the saved copy for that day.
- `site/archive/index.html` is the archive index.

If a generated section fails, the script logs the error and reuses the last good section with a small badge. A bad API call should not publish a broken or empty page.

## One-Time GitHub Setup

1. Create a repository named `daily-brief`.
2. Put these project files in that repository and push them to `main`.
3. In GitHub, open the repository and go to **Settings -> Secrets and variables -> Actions**.
4. Add a repository secret named `ANTHROPIC_API_KEY` with your Anthropic API key.
5. Go to **Settings -> Actions -> General** and confirm Actions are allowed.
6. Go to **Settings -> Pages** and set **Source** to **GitHub Actions**.
7. Run the **Daily Brief** workflow once manually from the **Actions** tab.

GitHub Pages will publish at:

`https://YOUR-GITHUB-USERNAME.github.io/daily-brief/`

For your connected GitHub account, the expected URL format is:

`https://griffinkosonocky.github.io/daily-brief/`

## Manual Rebuild from the GitHub Mobile App

1. Open the repository in the GitHub mobile app.
2. Tap **Actions**.
3. Tap **Daily Brief**.
4. Tap **Run workflow**.

To refresh only the bottom headline ticker, run the **News Ticker** workflow instead.

## Schedules

- Daily edition: `.github/workflows/daily.yml`
- Ticker refresh: `.github/workflows/ticker.yml`

The daily workflow runs at `10:30 UTC`, which is about `5:30 AM` in Chicago during standard time. Daylight saving time shifts this by an hour. GitHub scheduled jobs can also start a few minutes late.

To change the schedule, edit the `cron` line in the workflow file.

## Adding or Editing Historical Topics

Open `topics.json` and add a new object:

```json
{
  "title": "Your topic title",
  "era": "Modern",
  "region": "South America",
  "used_on": null
}
```

Leave `used_on` as `null` for unused topics. The generator marks topics as used after a successful article generation. When fewer than 10 unused topics remain, it asks Claude to append 30 fresh non-duplicate topics.

## Local Preview

Run:

```bash
python generate.py
```

Without `ANTHROPIC_API_KEY`, the script creates a complete local preview with clearly labeled sample content. In GitHub Actions, add the secret so the real generated content is used.

To test the market-card fallback:

```bash
SIMULATE_MARKETS_FAILURE=1 python generate.py
```

The page should still build and show the prior market card with a badge.

## Cost Estimate

The default models are:

- Historical read: `claude-sonnet-4-6`
- Markets: `claude-sonnet-4-6` with web search
- Trivia and topic refresh: `claude-haiku-4-5`

At current Anthropic API pricing, Sonnet is listed at about `$3 / million input tokens` and `$15 / million output tokens`, Haiku at about `$1 / million input tokens` and `$5 / million output tokens`, and web search at `$10 / 1,000 searches`.

A normal month should usually land around `$3-$8`, depending mostly on how many web searches the markets card uses and how long the historical reads are. Topic refreshes are rare and should add only a small amount.

## Privacy and Auth Note

GitHub Pages sites are public. The URL is not very discoverable, but it is not password-protected. This project should only publish non-sensitive content.

If you ever want authentication, the clean upgrade path is moving hosting to Cloudflare Pages and putting Cloudflare Access in front of it.
