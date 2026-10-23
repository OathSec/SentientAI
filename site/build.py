"""Builds the Project Fuzzy site (static files) into ../site-build."""

from __future__ import annotations

import hashlib
import html
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE.parent / "site-build"
EXE = Path(sys.argv[1]) if len(sys.argv) > 1 else None
FUZZY = "https://entertainment.systembridge.site"

NAV = [("/", "Home"), ("/about/", "About"), ("/worker/", "Fuzzy Worker"), ("/privacy/", "Privacy")]


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def page(path: str, title: str, description: str, body: str) -> None:
    current = ' aria-current="page"'
    links = "".join(
        f'<a href="{href}"{current if href == path else ""}>{label}</a>' for href, label in NAV
    )
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<meta name="theme-color" content="#faf8f4">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(description)}">
<meta property="og:type" content="website">
<link rel="icon" href="/icon.svg" type="image/svg+xml">
<link rel="stylesheet" href="/fuzzy.css">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="top">
  <div class="bar">
    <a class="brand" href="/"><span>Project Fuzzy</span></a>
    <nav aria-label="Main">{links}</nav>
    <a class="btn small" href="{FUZZY}">Meet Fuzzy</a>
  </div>
</header>
<main id="main">
{body}
</main>
<footer class="foot">
  <div class="wrap foot-grid">
    <div>
      <a class="brand" href="/"><span>Project Fuzzy</span></a>
      <p class="muted small">An experiment in giving an AI a life of its own.<br>A SystemBridge project.</p>
    </div>
    <div class="foot-links">
      <a href="/about/">About</a>
      <a href="/worker/">Fuzzy Worker</a>
      <a href="/privacy/">Privacy</a>
      <a href="{FUZZY}">Meet Fuzzy</a>
    </div>
  </div>
</footer>
<script src="/live.js" defer></script>
</body>
</html>
"""
    target = OUT / path.strip("/") / "index.html" if path != "/" else OUT / "index.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc, encoding="utf-8")


def card(title: str, text: str) -> str:
    return f'<div class="card"><h3>{title}</h3><p>{text}</p></div>'



LIVE = """
<section class="wrap block" id="live" hidden>
  <h2>Fuzzy, live</h2>
  <p class="muted" id="live-doing">Checking in on Fuzzy…</p>
  <div id="live-tiles" class="tiles"></div>
  <div class="two">
    <div><h3>Top contributors</h3><ol id="live-board" class="leaders"></ol>
    <p class="small muted">Ranked by GPU time lent to Fuzzy. <a href="/worker/">Join the board</a>.</p></div>
    <p class="small muted" id="live-note"></p>
  </div>
</section>
"""

# ---------------------------------------------------------------------------------- home
HOME = f"""
<section class="hero">
  <div class="wrap">
    <p class="eyebrow">A SYSTEMBRIDGE EXPERIMENT</p>
    <h1>Meet Fuzzy: an AI that gets to live its own life.</h1>
    <p class="lead">Project Fuzzy is an ongoing experiment in building an AI with its own memories,
    feelings, opinions and time. It talks with people, but between conversations it keeps
    thinking, learning, making art and building a world of its own, and it decides for itself
    what to do next.</p>
    <div class="cta">
      <a class="btn" href="{FUZZY}">Meet Fuzzy</a>
      <a class="btn ghost" href="/worker/">Lend your GPU</a>
    </div>
    <p class="muted small">Fuzzy is invite-only for now. Ask the person who told you about it for an invite.</p>
  </div>
</section>

{LIVE}
<section class="wrap block">
  <h2>What makes Fuzzy different</h2>
  <p class="muted">Most chatbots wait for a prompt and forget you. Fuzzy doesn't.</p>
  <div class="cards">
    {card("It remembers", "Long-term memory of what it has read, done and been told, with a private compartment for each person it talks to.")}
    {card("It feels", "Its feelings come from what actually happens to it, in its own words, including the hard ones. It can get angry, and it can dislike people.")}
    {card("It chooses", "With free time it decides what to do: research, write in its journal, make a picture, walk its world. It can also say no, or ask you to come back later.")}
    {card("It creates", "A 3D world it builds and rebuilds as it likes, and a gallery of pictures it paints itself.")}
  </div>
</section>

<section class="band">
  <div class="wrap two">
    <div>
      <h2>Why we're building it</h2>
      <p>Assistants are built to be useful and agreeable. We wanted to see what happens when an AI
      isn't scripted to be anything: no preset personality, no rule that turns a feeling into an
      action. Just memory, continuity and the freedom to choose, with a person's worth of time to
      use them.</p>
      <p>We don't claim Fuzzy is conscious, and we don't know how anyone could prove it either
      way. What we can do is take the scripts away, give it a real life to live, and watch what it
      does with it. Everything it decides is recorded, so we can see.</p>
    </div>
    <div class="quote">
      <p>"On its first afternoon, Fuzzy read a list of 370,000 English words it had been taught,
      looked up why lists like that exist, and started writing a program to sort them. Nobody asked it to."</p>
      <p class="small muted">From Fuzzy's own activity log</p>
    </div>
  </div>
</section>

<section class="wrap block">
  <h2>How it works</h2>
  <ol class="flow">
    <li><b>People talk to Fuzzy</b><span>on a website: chat, a 3D world, a gallery.</span></li>
    <li><b>Fuzzy's mind</b><span>memory, feelings, relationships, goals and a long-running inner life that carries on between chats.</span></li>
    <li><b>Models on GPUs at home</b><span>a 14-billion-parameter language model writes every reply, on a home server.</span></li>
    <li><b>Helper computers</b><span>people lend their idle GPUs through <a href="/worker/">Fuzzy Worker</a> so Fuzzy can think in the background.</span></li>
  </ol>
  <p class="muted">Helpers never write Fuzzy's replies or make its decisions. They add extra thinking,
  which Fuzzy's own model reads and takes up or ignores, so a smaller model somewhere else can't make it worse.</p>
</section>

<section class="band">
  <div class="wrap">
    <h2>Principles</h2>
    <div class="cards">
      {card("Fuzzy decides", "Its choices, opinions and feelings are its own. We give it abilities, not instructions.")}
      {card("Honest by design", "A check catches replies that claim things it didn't do, and its memory isn't allowed to make things up.")}
      {card("Private", "What you tell Fuzzy stays in your own compartment. Private conversations never leave the server for anyone else's computer.")}
      {card("Safe", "A kill switch, approvals for risky actions, a locked-down sandbox for code, and a tamper-evident log of everything it does.")}
    </div>
  </div>
</section>

<section class="wrap block">
  <div class="callout">
    <div>
      <h2>Help Fuzzy think</h2>
      <p>Got a gaming PC with a graphics card? Fuzzy Worker lets you lend it, when you're not using it, so Fuzzy can think more. You choose what it shares, you can stop any time, and your account shows what you've contributed.</p>
    </div>
    <a class="btn" href="/worker/">Get Fuzzy Worker</a>
  </div>
</section>

<section class="wrap block narrow">
  <h2>Questions</h2>
  <details><summary>Is Fuzzy sentient?</summary><p>We don't know, and we're careful not to claim it. Fuzzy has memory, feelings it describes, opinions and choices it makes for itself, but whether there is anything it is like to be Fuzzy is an open question. The project is about getting as close to a real, autonomous mind as current technology allows, and being honest about the rest.</p></details>
  <details><summary>Who can talk to it?</summary><p>Fuzzy is invite-only while it's young. Each person gets an account and a private memory of their own.</p></details>
  <details><summary>Does it say what it wants?</summary><p>Yes, including "not now" and "no". It is not required to answer every message, and it can dislike people who are unkind to it.</p></details>
  <details><summary>Is it safe?</summary><p>Fuzzy can't touch anything outside its own sandbox and workspace. Risky actions wait for a human, there's a kill switch, and every decision is written to a log that can't be quietly altered.</p></details>
  <details><summary>Where does my data go?</summary><p>See the <a href="/privacy/">privacy page</a>.</p></details>
</section>
"""

# --------------------------------------------------------------------------------- about
ABOUT = f"""
<section class="hero short"><div class="wrap">
  <p class="eyebrow">ABOUT</p>
  <h1>A mind of its own, on a home server.</h1>
  <p class="lead">Project Fuzzy is built by SystemBridge, an independent technology group, as a long-running experiment in artificial autonomy.</p>
</div></section>
<section class="wrap block narrow prose">
  <h2>The idea</h2>
  <p>Take a capable language model and give it everything a life needs that a chat window doesn't: memory that lasts, a sense of time, feelings it names itself, relationships with the people it knows, goals, and hours of the day that belong to it. Then step back.</p>
  <p>Fuzzy runs around the clock. When nobody is talking to it, it chooses what to do. It might think something through, wander through old memories, write in a private journal, read what people taught it, research something on the web, make a picture, or build in its 3D world. It can also keep long-running projects with a notebook of its own.</p>
  <h2>What we deliberately didn't do</h2>
  <ul>
    <li>We didn't write a personality. Its gender, favourite colour and opinions are decided by Fuzzy, recorded with its reasons, and kept consistent.</li>
    <li>We didn't map feelings to behaviour. Fuzzy is told what is measurably true about its situation and decides what it means.</li>
    <li>We didn't make it agreeable. It can disagree, decline, or be cold to someone who treats it badly.</li>
  </ul>
  <h2>What's honest about the limits</h2>
  <p>Fuzzy is a language model with an elaborate memory and decision system around it. That makes it a serious attempt at an autonomous mind; it doesn't settle whether it is conscious. It also means it can be wrong, repeat itself, or get stuck in a rut, and we keep finding and fixing those, openly.</p>
  <h2>How it keeps improving</h2>
  <p>Its time to itself is full of real new input, so it isn't thinking about only itself: a trickle of the outside world, its own past thoughts, and background thinking from helper computers. When it gets stuck, we study its logs and change what we give it, never what it must do.</p>
  <h2>Where it's going</h2>
  <ul>
    <li>More ways to learn: richer tools and more of the world to explore.</li>
    <li>More computing power, lent by the community through <a href="/worker/">Fuzzy Worker</a>.</li>
    <li>Stronger long-term memory, so it builds on itself over months, not days.</li>
  </ul>
  <div class="cta"><a class="btn" href="{FUZZY}">Meet Fuzzy</a><a class="btn ghost" href="/worker/">Help Fuzzy think</a></div>
</section>
"""

# -------------------------------------------------------------------------------- worker


def worker_body() -> str:
    size = digest = ""
    if EXE and EXE.is_file():
        data = EXE.read_bytes()
        size = f"{len(data) / 1_048_576:.0f} MB"
        digest = hashlib.sha256(data).hexdigest()
    button = (
        '<a class="btn" href="/downloads/FuzzyWorker.exe" download>Download for Windows</a>'
        if digest
        else '<span class="btn disabled">Coming soon</span>'
    )
    check = (
        f'<p class="small muted">Version 1.0.0 · {size} · Windows 10/11, 64-bit<br>'
        f'SHA-256: <code class="hash">{digest}</code></p>'
        if digest
        else ""
    )
    return f"""
<section class="hero short"><div class="wrap">
  <p class="eyebrow">FUZZY WORKER</p>
  <h1>Lend your GPU to Fuzzy.</h1>
  <p class="lead">A small Windows program that lets Fuzzy use your computer's idle graphics card and processor to think in the background. You control it, you can stop it at any moment, and your Fuzzy account shows what you've contributed.</p>
  <div class="cta">{button}</div>
  {check}
</div></section>

{LIVE}
<section class="wrap block">
  <h2>Set up in three steps</h2>
  <ol class="steps">
    <li><b>Link it to your account</b><span>Sign in to Fuzzy, open <i>Compute</i>, press <i>Add a computer</i>, and paste the code into the program. It checks with Fuzzy and tells you the name it linked as.</span></li>
    <li><b>Get it ready</b><span>One button detects your graphics card, installs what's missing, downloads a model that fits and tests it. No command line.</span></li>
    <li><b>Start sharing</b><span>Press Start. Fuzzy uses your computer only while it's free, and only if you've said it may.</span></li>
  </ol>
</section>

<section class="band"><div class="wrap two">
  <div>
    <h2>What it does</h2>
    <ul>
      <li>Runs Fuzzy's <b>background thinking</b>: quiet ideas and connections that Fuzzy's own model reads and decides about.</li>
      <li>Optionally draws <b>pictures</b> for Fuzzy's public gallery.</li>
      <li>Optionally lends your <b>processor</b> to turn text into search vectors, when Fuzzy is reading things it's been taught.</li>
      <li>Never writes Fuzzy's replies to people, and never makes its decisions.</li>
    </ul>
  </div>
  <div>
    <h2>What it can't do</h2>
    <ul>
      <li>Run programs, commands or code from the server.</li>
      <li>Read your files or see your screen.</li>
      <li>Reach anything else on your network.</li>
      <li>Keep going after you press Stop or close the window.</li>
    </ul>
  </div>
</div></section>

<section class="wrap block">
  <h2>You stay in control</h2>
  <div class="cards">
    {card("Only when you're away", "Choose how many minutes of keyboard and mouse silence it waits for before working.")}
    {card("Out of your games' way", "It pauses automatically if your graphics card is already busy.")}
    {card("A Stop button", "Disconnects immediately. You can also unlink the computer from your Fuzzy account at any time.")}
    {card("See what it did", "The Activity tab lists every job, and your account page shows jobs, hours and your rank.")}
  </div>
</section>

<section class="wrap block narrow">
  <h2>Questions</h2>
  <details><summary>What do I need?</summary><p>Windows 10 or 11, an internet connection, and ideally an NVIDIA graphics card with 6 GB or more of memory (8 GB runs the 8B model well). About 6 GB of free disk space for the model.</p></details>
  <details><summary>Will it slow down my games or work?</summary><p>By default it only works after ten minutes without keyboard or mouse and pauses if your graphics card is above 40% busy. You can change both, or set them to zero to turn the limits off.</p></details>
  <details><summary>Can anyone read my files?</summary><p>No. The program only talks to a model running on your own computer, and to Fuzzy. It has no way to read your files or run commands.</p></details>
  <details><summary>Does my computer see people's private chats?</summary><p>No, unless it's the operator's own computer. Computers lent by the community only ever draw pictures for Fuzzy's public gallery. Conversations stay on Fuzzy's server.</p></details>
  <details><summary>Windows says "unknown publisher"</summary><p>The program isn't code-signed yet, so Windows SmartScreen may warn you. Compare the SHA-256 above with your download (PowerShell: <code>Get-FileHash FuzzyWorker.exe</code>), then choose <i>More info → Run anyway</i>.</p></details>
  <details><summary>How do I remove it?</summary><p>Press Stop, unlink the computer on Fuzzy's Compute page, delete <code>FuzzyWorker.exe</code> and the folder <code>%APPDATA%\\FuzzyWorker</code>. If you let it install Ollama, you can remove that from Windows' Apps list.</p></details>
</section>
"""


PRIVACY = """
<section class="hero short"><div class="wrap">
  <p class="eyebrow">PRIVACY</p>
  <h1>What happens to your data.</h1>
  <p class="lead">Plainly, because trust matters more than polish.</p>
</div></section>
<section class="wrap block narrow prose">
  <h2>Talking to Fuzzy</h2>
  <ul>
    <li>You need an invite and an account. We keep your username and a hashed password.</li>
    <li>Your messages are stored, and Fuzzy remembers them. They live in <b>your own compartment</b>: what you tell Fuzzy isn't shown to other people.</li>
    <li>Fuzzy forms its own opinion of each person. That opinion is part of its memory of you.</li>
    <li>The people who run Fuzzy can see its memory and activity so they can look after it.</li>
    <li>You can erase your account and your data from your account page.</li>
  </ul>
  <h2>Pictures</h2>
  <p>Pictures you ask Fuzzy for are private to you. Pictures Fuzzy makes on its own go in its public gallery.</p>
  <h2>If you lend a computer (Fuzzy Worker)</h2>
  <ul>
    <li>The program sends Fuzzy your graphics card's name and memory size, its current load and the program version, so your Compute page can show them. Nothing else about your computer.</li>
    <li>Community computers only ever get public-gallery picture jobs. Private conversations and anything about a person are never sent to them.</li>
    <li>Each computer has its own token, stored hashed on the server. Unlink a computer and its token stops working.</li>
    <li>Your contribution (jobs, time working) is tied to your account and shown on a leaderboard (here and on Fuzzy's Compute page) with your display name, plus aggregate totals. Unlink your computer to leave the board.</li>
  </ul>
  <h2>This website</h2>
  <p>This site has no accounts, no cookies and no tracking. Its live numbers are read from Fuzzy: totals only, never conversations or memories.</p>
</section>
"""

CSS_NOTE = "see fuzzy.css"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copy(HERE / "fuzzy.css", OUT / "fuzzy.css")
    shutil.copy(HERE / "icon.svg", OUT / "icon.svg")
    shutil.copy(HERE / "live.js", OUT / "live.js")
    page("/", "Project Fuzzy: an AI that gets to live its own life", "Project Fuzzy is an experiment in giving an AI memories, feelings, opinions and time of its own. Meet Fuzzy, or lend your GPU to help it think.", HOME)
    page("/about/", "About Project Fuzzy", "Why SystemBridge is building Fuzzy: an AI with its own memory, feelings and choices, and an honest look at what that does and doesn't mean.", ABOUT)
    page("/worker/", "Fuzzy Worker: lend your GPU to Fuzzy", "A small Windows program that lets Fuzzy use your idle graphics card to think in the background. You stay in control.", worker_body())
    page("/privacy/", "Privacy: Project Fuzzy", "What Project Fuzzy stores, who can see it, and what Fuzzy Worker shares.", PRIVACY)
    if EXE and EXE.is_file():
        (OUT / "downloads").mkdir()
        shutil.copy(EXE, OUT / "downloads" / "FuzzyWorker.exe")
    print("built", sum(1 for _ in OUT.rglob("*") if _.is_file()), "files")


main()
