(function () {
  "use strict";

  const SOURCE = "js/opencv.js?v=1";
  const TIMEOUT_MS = 120000;
  let settled = false;
  let timeoutId = 0;
  let resolveReady;
  let rejectReady;

  function dispatch(name, detail) {
    window.dispatchEvent(new CustomEvent(name, { detail: detail || {} }));
  }

  function isReady(cv) {
    return !!(
      cv &&
      typeof cv.Mat === "function" &&
      typeof cv.imread === "function" &&
      typeof cv.cvtColor === "function"
    );
  }

  function finish(cv) {
    if (settled) return;
    const readyCv = cv || window.cv;
    if (!isReady(readyCv)) return;

    settled = true;
    window.clearTimeout(timeoutId);
    window.cv = readyCv;
    dispatch("ppai-opencv-ready", { source: SOURCE });

    // IMPORTANT: this OpenCV build is thenable but not a normal Promise.
    // Never resolve with cv itself or `await window.cv`.
    resolveReady({ ready: true, source: SOURCE });
  }

  function fail(errorLike) {
    if (settled) return;
    settled = true;
    window.clearTimeout(timeoutId);
    const error = errorLike instanceof Error
      ? errorLike
      : new Error(String(errorLike || "OpenCV failed to initialize."));
    dispatch("ppai-opencv-error", {
      message: error.message,
      source: SOURCE
    });
    rejectReady(error);
  }

  function registerRuntimeCallback() {
    const cv = window.cv;

    if (!cv) {
      fail(new Error("js/opencv.js loaded, but window.cv was not created."));
      return;
    }

    if (isReady(cv)) {
      finish(cv);
      return;
    }

    // Known-good callback pattern from the working FreeCell scanner.
    // Do not `await cv`; this build's custom then() is callback-style.
    if (typeof cv.then === "function") {
      try {
        cv.then(function (readyCv) {
          finish(readyCv || window.cv);
        });
      } catch (error) {
        fail(error);
      }
    }

    // Safari/iPhone defensive polling.
    (function poll() {
      if (settled) return;
      if (isReady(window.cv)) {
        finish(window.cv);
        return;
      }
      window.setTimeout(poll, 100);
    }());
  }

  function createReadyPromise() {
    settled = false;
    window.ppaiCvReady = new Promise(function (resolve, reject) {
      resolveReady = resolve;
      rejectReady = reject;
    });

    dispatch("ppai-opencv-loading", { source: SOURCE, attempt: 1 });
    timeoutId = window.setTimeout(function () {
      fail(new Error(
        "OpenCV did not finish initializing. On Safari, fully close and reopen Safari, then try again."
      ));
    }, TIMEOUT_MS);

    registerRuntimeCallback();
    return window.ppaiCvReady;
  }

  window.ppaiCvRetry = function () {
    // Re-register against the existing OpenCV object rather than reloading
    // the large library inside a live Safari page.
    return createReadyPromise();
  };

  createReadyPromise();
}());
