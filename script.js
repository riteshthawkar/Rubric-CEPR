const tabs = Array.from(document.querySelectorAll('.tab'));
const panels = Array.from(document.querySelectorAll('.analysis-figure'));

tabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    const target = tab.dataset.figure;
    tabs.forEach((item) => {
      const active = item === tab;
      item.classList.toggle('is-active', active);
      item.setAttribute('aria-selected', active ? 'true' : 'false');
    });
    panels.forEach((panel) => {
      panel.classList.toggle('is-hidden', panel.dataset.panel !== target);
    });
  });
});

const copyButton = document.querySelector('[data-copy-target]');
if (copyButton) {
  copyButton.addEventListener('click', async () => {
    const target = document.getElementById(copyButton.dataset.copyTarget);
    const text = target ? target.innerText.trim() : '';
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      const original = copyButton.textContent;
      copyButton.textContent = 'Copied';
      window.setTimeout(() => {
        copyButton.textContent = original;
      }, 1400);
    } catch (error) {
      copyButton.textContent = 'Select BibTeX';
    }
  });
}

// ---------- Style "base -> ours" table deltas ----------
const deltaPattern = /^\s*(-?\d+(?:\.\d+)?)\s*->\s*(-?\d+(?:\.\d+)?)\s*$/;
document.querySelectorAll('.table-panel td').forEach((cell) => {
  const match = cell.textContent.match(deltaPattern);
  if (!match) return;
  const diff = parseFloat(match[2]) - parseFloat(match[1]);
  const rounded = Math.round(diff * 10) / 10;
  const sign = rounded > 0 ? '+' : '';
  cell.classList.add('delta');
  cell.innerHTML =
    `${match[1]}<span class="arrow">&rarr;</span><b>${match[2]}</b>` +
    (rounded !== 0 ? ` <span class="delta-chip">${sign}${rounded}</span>` : '');
});

// ---------- Scroll reveal ----------
const revealEls = Array.from(document.querySelectorAll('.reveal'));
if ('IntersectionObserver' in window && revealEls.length) {
  const revealObserver = new IntersectionObserver(
    (entries, obs) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          obs.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  revealEls.forEach((el) => revealObserver.observe(el));
} else {
  revealEls.forEach((el) => el.classList.add('is-visible'));
}

// ---------- Animate result bars when in view ----------
const bars = Array.from(document.querySelectorAll('.bar-fill'));
if ('IntersectionObserver' in window && bars.length) {
  const barObserver = new IntersectionObserver(
    (entries, obs) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const bar = entry.target;
          bar.style.width = `${bar.dataset.width}%`;
          obs.unobserve(bar);
        }
      });
    },
    { threshold: 0.4 }
  );
  bars.forEach((bar) => barObserver.observe(bar));
} else {
  bars.forEach((bar) => {
    bar.style.width = `${bar.dataset.width}%`;
  });
}

// ---------- Active section in sticky nav ----------
const navLinks = Array.from(document.querySelectorAll('.section-nav a'));
const navTargets = navLinks
  .map((link) => document.querySelector(link.getAttribute('href')))
  .filter(Boolean);

if ('IntersectionObserver' in window && navTargets.length) {
  const navObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          navLinks.forEach((link) => {
            link.classList.toggle('is-current', link.getAttribute('href') === `#${id}`);
          });
        }
      });
    },
    { rootMargin: '-45% 0px -50% 0px', threshold: 0 }
  );
  navTargets.forEach((target) => navObserver.observe(target));
}
