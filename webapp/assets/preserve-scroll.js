(() => {
  let pendingObserver = null;
  let restoreGeneration = 0;

  function capturePosition(pageContent) {
    const scrollY = window.scrollY;
    const headerHeight =
      document.querySelector(".site-header")?.getBoundingClientRect().height || 0;
    const sections = Array.from(
      pageContent.querySelectorAll("main section[data-analysis-section]")
    );
    const focusY = headerHeight + (window.innerHeight - headerHeight) * 0.45;
    let sectionIndex = sections.findIndex((section) => {
      const bounds = section.getBoundingClientRect();
      return bounds.top <= focusY && bounds.bottom > focusY;
    });

    if (sectionIndex < 0) {
      sectionIndex = sections.findIndex(
        (section) => section.getBoundingClientRect().bottom > focusY
      );
    }
    if (sectionIndex < 0 && sections.length) {
      sectionIndex = sections.length - 1;
    }

    const section = sections[sectionIndex];
    return {
      scrollY,
      sectionIndex,
      sectionOffset: section ? -section.getBoundingClientRect().top : 0,
    };
  }

  function restorePosition(pageContent, position) {
    const sections = Array.from(
      pageContent.querySelectorAll("main section[data-analysis-section]")
    );
    const section = sections[position.sectionIndex];
    let targetY = position.scrollY;

    if (section) {
      const sectionTop = window.scrollY + section.getBoundingClientRect().top;
      targetY = sectionTop + position.sectionOffset;
    }

    const maximumY = Math.max(
      0,
      document.documentElement.scrollHeight - window.innerHeight
    );
    window.scrollTo({ top: Math.min(Math.max(targetY, 0), maximumY), behavior: "instant" });
  }

  document.addEventListener(
    "click",
    (event) => {
      const target = event.target;
      if (!(target instanceof Element) || !target.closest("#confidence-select")) {
        return;
      }

      const pageContent = document.getElementById("page-content");
      if (!pageContent) {
        return;
      }

      const position = capturePosition(pageContent);
      const generation = ++restoreGeneration;
      const pathname = window.location.pathname;
      pendingObserver?.disconnect();
      const observer = new MutationObserver(() => {
        observer.disconnect();
        if (pendingObserver === observer) {
          pendingObserver = null;
        }
        [0, 100, 350, 800, 1600].forEach((delay) => {
          window.setTimeout(() => {
            if (
              generation === restoreGeneration &&
              pathname === window.location.pathname
            ) {
              window.requestAnimationFrame(() => restorePosition(pageContent, position));
            }
          }, delay);
        });
      });
      pendingObserver = observer;
      pendingObserver.observe(pageContent, { childList: true, subtree: true });

      window.setTimeout(() => {
        observer.disconnect();
        if (pendingObserver === observer) {
          pendingObserver = null;
        }
      }, 5000);
    },
    true
  );
})();
