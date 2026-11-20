import { Client, handle_file } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

const HF_SPACE_ID = "WilliamJovaski/Twizzle";
const SAMPLE_VIDEO_URL = "/static/contoh.mp4";
const ELEMENTS_DATA_URL = "/static/elements.json";
const SKATERS_DATA_URL = "/static/skaters.json";

const GRADIO_MODE_LABELS = {
  fast: "Fast (X3D, cepat)",
  accurate: "Accurate (Fused X3D + R(2+1)D, lebih akurat, lebih lambat)",
};

document.addEventListener("DOMContentLoaded", () => {
  // =========================================================================
  // REUSABLE COMPONENT: SectionEyebrow (props: number, text)
  // =========================================================================
  function SectionEyebrow({ number, text }) {
    const row = document.createElement("div");
    row.className = "section-eyebrow-row";

    const numEl = document.createElement("span");
    numEl.className = "section-eyebrow-num";
    numEl.textContent = String(number || "").replace(/^\/\s*/, "").trim();

    const divider = document.createElement("span");
    divider.className = "section-eyebrow-divider";
    divider.setAttribute("aria-hidden", "true");

    const textEl = document.createElement("span");
    textEl.className = "section-eyebrow-text";
    textEl.textContent = text || "";

    const bar = document.createElement("span");
    bar.className = "section-eyebrow-bar";
    bar.setAttribute("aria-hidden", "true");

    row.appendChild(numEl);
    row.appendChild(divider);
    row.appendChild(textEl);

    const frag = document.createDocumentFragment();
    frag.appendChild(row);
    frag.appendChild(bar);
    return frag;
  }

  document.querySelectorAll(".section-eyebrow[data-eyebrow-num]").forEach((el) => {
    el.replaceChildren(
      SectionEyebrow({
        number: el.dataset.eyebrowNum,
        text: el.dataset.eyebrowText,
      })
    );
  });

  // =========================================================================
  // 0. HERO CHOREOGRAPHY (UNTOUCHED)
  // =========================================================================
  const heroStage = document.querySelector(".hero-stage");
  const heroGiantText = document.getElementById("heroGiantText");
  const heroTopBar = document.getElementById("heroTopBar");
  const replayHeroBtn = document.getElementById("replayHeroBtn");

  let animTimers = [];

  function clearAnimTimers() {
    animTimers.forEach((t) => clearTimeout(t));
    animTimers = [];
  }

  function runHeroChoreography() {
    clearAnimTimers();

    heroStage.classList.remove("stage-enter", "stage-color", "stage-figures", "stage-float");
    heroGiantText.classList.remove("text-in", "text-out");
    heroTopBar.classList.remove("visible");

    void heroStage.offsetWidth;

    animTimers.push(
      setTimeout(() => {
        heroStage.classList.add("stage-enter", "stage-color");
      }, 60)
    );

    animTimers.push(
      setTimeout(() => {
        heroGiantText.classList.add("text-in");
      }, 1750)
    );

    animTimers.push(
      setTimeout(() => {
        heroGiantText.classList.add("text-out");
        heroStage.classList.add("stage-figures");
      }, 3400)
    );

    animTimers.push(
      setTimeout(() => {
        heroTopBar.classList.add("visible");
      }, 4150)
    );

    animTimers.push(
      setTimeout(() => {
        heroStage.classList.add("stage-float");
      }, 4900)
    );
  }

  runHeroChoreography();

  if (replayHeroBtn) {
    replayHeroBtn.addEventListener("click", () => {
      window.scrollTo({ top: 0, behavior: "smooth" });
      runHeroChoreography();
    });
  }

  // Smooth Lazy Scroll-Reveal for Sections 01-06
  const revealSections = document.querySelectorAll(".light-surface-root .editorial-section");
  if ("IntersectionObserver" in window && revealSections.length > 0) {
    const secObserver = new IntersectionObserver(
      (entries, observer) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { rootMargin: "0px 0px -60px 0px", threshold: 0.04 }
    );
    revealSections.forEach((sec) => secObserver.observe(sec));
  } else {
    revealSections.forEach((sec) => sec.classList.add("is-visible"));
  }

  // =========================================================================
  // 1. SECTION 01: 11 ELEMENTS & MOVEMENTS EXPLORER
  // =========================================================================
  const elementsTabList = document.getElementById("elementsTabList");
  const elementPreviewVideo = document.getElementById("elementPreviewVideo");
  const elementDiagramOverlay = document.getElementById("elementDiagramOverlay");
  const elPreviewIndex = document.getElementById("elPreviewIndex");
  const elPreviewCategory = document.getElementById("elPreviewCategory");
  const elPreviewCode = document.getElementById("elPreviewCode");
  const elPreviewName = document.getElementById("elPreviewName");
  const elPreviewDifficulty = document.getElementById("elPreviewDifficulty");
  const elPreviewSummary = document.getElementById("elPreviewSummary");
  const elPreviewTakeoff = document.getElementById("elPreviewTakeoff");
  const elPreviewLanding = document.getElementById("elPreviewLanding");
  const elPreviewCue = document.getElementById("elPreviewCue");

  let elementsList = [];
  let activeElementIdx = 0;
  let previewVideoLoaded = false;

  // Lazy-load the preview video only when #elements scrolls into view
  const elementsSection = document.getElementById("elements");
  if ("IntersectionObserver" in window && elementsSection) {
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !previewVideoLoaded && elementsList.length > 0) {
            previewVideoLoaded = true;
            const item = elementsList[activeElementIdx];
            if (item && item.previewVideo) {
              elementPreviewVideo.src = item.previewVideo;
              elementPreviewVideo.play().catch(() => {});
            }
          }
        });
      },
      { rootMargin: "160px" }
    );
    obs.observe(elementsSection);
  }

  function buildDiagramSvg(item) {
    const cat = item.category;
    if (cat === "Jump") {
      return `
        <svg viewBox="0 0 520 190" fill="none" xmlns="http://www.w3.org/2000/svg">
          <line x1="24" y1="152" x2="496" y2="152" stroke="rgba(248,250,252,0.28)" stroke-width="1.5"/>
          <path d="M 42 152 C 105 152, 135 144, 175 132" stroke="#CBD5E1" stroke-width="2" stroke-dasharray="5 5"/>
          <path d="M 175 132 C 230 34, 310 34, 365 132" stroke="#2997FF" stroke-width="3.2"/>
          <path d="M 365 132 C 400 148, 440 152, 480 152" stroke="#F8FAFC" stroke-width="2.4"/>
          <circle cx="175" cy="132" r="5.5" fill="#FF9500"/>
          <circle cx="270" cy="56" r="5.5" fill="#2997FF"/>
          <circle cx="365" cy="132" r="5.5" fill="#22C55E"/>
          <text x="122" y="176" fill="#F8FAFC" font-size="14.5" font-weight="700" font-family="Plus Jakarta Sans, Inter, sans-serif">Takeoff (${escapeHtml(item.code)})</text>
          <text x="270" y="34" text-anchor="middle" fill="#FFFFFF" font-size="15" font-weight="700" font-family="Plus Jakarta Sans, Inter, sans-serif">${escapeHtml(item.name)} Apex</text>
          <text x="330" y="176" fill="#F8FAFC" font-size="14.5" font-weight="700" font-family="Plus Jakarta Sans, Inter, sans-serif">Landing Outflow</text>
        </svg>
      `;
    }
    if (cat === "Spin") {
      return `
        <svg viewBox="0 0 520 190" fill="none" xmlns="http://www.w3.org/2000/svg">
          <line x1="260" y1="26" x2="260" y2="168" stroke="rgba(248,250,252,0.3)" stroke-width="1.5" stroke-dasharray="4 4"/>
          <ellipse cx="260" cy="98" rx="92" ry="26" stroke="#2997FF" stroke-width="2.8"/>
          <ellipse cx="260" cy="132" rx="58" ry="16" stroke="#FF9500" stroke-width="2.2" stroke-dasharray="6 4"/>
          <circle cx="260" cy="98" r="5.5" fill="#22C55E"/>
          <text x="276" y="42" fill="#FFFFFF" font-size="15" font-weight="700" font-family="Plus Jakarta Sans, Inter, sans-serif">Spin Axis (${escapeHtml(item.code)})</text>
          <text x="364" y="103" fill="#F8FAFC" font-size="14.5" font-weight="700" font-family="Plus Jakarta Sans, Inter, sans-serif">Centered Rotation</text>
        </svg>
      `;
    }
    return `
      <svg viewBox="0 0 520 190" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="34" y="40" width="452" height="118" rx="59" stroke="rgba(248,250,252,0.28)" stroke-width="1.5"/>
        <path d="M 70 99 C 135 42, 195 156, 260 99 C 325 42, 385 156, 450 99" stroke="#2997FF" stroke-width="3" stroke-dasharray="7 5"/>
        <circle cx="165" cy="99" r="5" fill="#FF9500"/>
        <circle cx="355" cy="99" r="5" fill="#22C55E"/>
        <text x="260" y="26" text-anchor="middle" fill="#FFFFFF" font-size="15" font-weight="700" font-family="Plus Jakarta Sans, Inter, sans-serif">Full-Rink Step / Turn Coverage (${escapeHtml(item.code)})</text>
      </svg>
    `;
  }

  function selectElement(idx) {
    if (!elementsList[idx]) return;
    activeElementIdx = idx;
    const item = elementsList[idx];

    const buttons = elementsTabList.querySelectorAll(".element-tab-btn");
    buttons.forEach((btn, i) => {
      btn.setAttribute("aria-selected", i === idx ? "true" : "false");
    });

    elPreviewIndex.textContent = `${item.index} / ${String(elementsList.length).padStart(2, "0")}`;
    elPreviewCategory.textContent = item.category;
    elPreviewCode.textContent = `ISU Code: ${item.code}`;
    elPreviewName.textContent = item.name;
    elPreviewDifficulty.textContent = item.difficulty;
    elPreviewSummary.textContent = item.summary;
    elPreviewTakeoff.textContent = item.takeoffEdge;
    elPreviewLanding.textContent = item.landingEdge;
    elPreviewCue.textContent = item.keyCue;
    elementDiagramOverlay.innerHTML = buildDiagramSvg(item);

    if (previewVideoLoaded && item.previewVideo) {
      if (elementPreviewVideo.getAttribute("src") !== item.previewVideo) {
        elementPreviewVideo.src = item.previewVideo;
      }
      elementPreviewVideo.play().catch(() => {});
    }
  }

  async function initElementsExplorer() {
    try {
      const res = await fetch(ELEMENTS_DATA_URL);
      if (!res.ok) throw new Error("Failed to load elements.json");
      elementsList = await res.json();

      elementsTabList.innerHTML = "";
      elementsList.forEach((item, idx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "element-tab-btn";
        btn.setAttribute("role", "tab");
        btn.setAttribute("aria-selected", idx === 0 ? "true" : "false");
        btn.innerHTML = `
          <span class="tab-left">
            <span class="tab-num">${item.index}</span>
            <span>${item.name}</span>
          </span>
          <span class="tab-cat">${item.category}</span>
        `;
        btn.addEventListener("click", () => selectElement(idx));
        elementsTabList.appendChild(btn);
      });

      selectElement(0);
    } catch (err) {
      elPreviewSummary.textContent = "Unable to load static/elements.json.";
    }
  }

  initElementsExplorer();

  // =========================================================================
  // 2. SECTION 02: AI CLASSIFIER STUDIO (Connected to HF Space)
  // =========================================================================
  const dropZone = document.getElementById("dropZone");
  const dropPrompt = document.getElementById("dropPrompt");
  const videoInput = document.getElementById("videoInput");
  const videoPreviewWrap = document.getElementById("videoPreviewWrap");
  const videoPreview = document.getElementById("videoPreview");
  const clearVideoBtn = document.getElementById("clearVideoBtn");
  const loadSampleBtn = document.getElementById("loadSampleBtn");
  const classifyBtn = document.getElementById("classifyBtn");
  const classifyBtnText = document.getElementById("classifyBtnText");
  const errorBox = document.getElementById("errorBox");

  const modeFastCard = document.getElementById("modeFastCard");
  const modeAccurateCard = document.getElementById("modeAccurateCard");
  const modeRadios = document.querySelectorAll('input[name="mode"]');

  const metaBadge = document.getElementById("metaBadge");
  const emptyState = document.getElementById("emptyState");
  const resultsContent = document.getElementById("resultsContent");
  const topClassName = document.getElementById("topClassName");
  const topClassCategory = document.getElementById("topClassCategory");
  const topConfidence = document.getElementById("topConfidence");
  const top3Bars = document.getElementById("top3Bars");
  const allClassesBody = document.getElementById("allClassesBody");
  const framesSection = document.getElementById("framesSection");
  const framesGallery = document.getElementById("framesGallery");

  let selectedFile = null;
  let usingSampleVideo = false;
  let gradioClientPromise = null;

  function getGradioClient() {
    if (!gradioClientPromise) {
      gradioClientPromise = Client.connect(HF_SPACE_ID).catch((err) => {
        gradioClientPromise = null;
        throw err;
      });
    }
    return gradioClientPromise;
  }

  const JUMPS = new Set(["Toeloop", "Salchow", "Flip", "Loop", "Lutz", "Axel"]);
  const SPINS = new Set(["CamelSpin", "SitSpin", "UprightSpin", "NoBasic"]);

  function getElementCategory(name) {
    if (JUMPS.has(name)) return "Jump Element";
    if (SPINS.has(name)) return "Spin Element";
    return "Step / Choreo Sequence";
  }

  function showError(msg) {
    if (!msg) {
      errorBox.classList.add("hidden");
      errorBox.textContent = "";
      return;
    }
    errorBox.textContent = msg;
    errorBox.classList.remove("hidden");
  }

  function setVideoPreviewFromFile(file) {
    selectedFile = file;
    usingSampleVideo = false;
    showError("");
    const url = URL.createObjectURL(file);
    videoPreview.src = url;
    dropPrompt.classList.add("hidden");
    videoPreviewWrap.classList.remove("hidden");
  }

  dropPrompt.addEventListener("click", () => videoInput.click());
  dropPrompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      videoInput.click();
    }
  });

  videoInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      setVideoPreviewFromFile(e.target.files[0]);
    }
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setVideoPreviewFromFile(e.dataTransfer.files[0]);
    }
  });

  clearVideoBtn.addEventListener("click", () => {
    selectedFile = null;
    usingSampleVideo = false;
    videoInput.value = "";
    videoPreview.pause();
    videoPreview.removeAttribute("src");
    videoPreviewWrap.classList.add("hidden");
    dropPrompt.classList.remove("hidden");
    showError("");
  });

  loadSampleBtn.addEventListener("click", () => {
    selectedFile = null;
    usingSampleVideo = true;
    showError("");
    videoPreview.src = SAMPLE_VIDEO_URL;
    dropPrompt.classList.add("hidden");
    videoPreviewWrap.classList.remove("hidden");
  });

  modeRadios.forEach((radio) => {
    radio.addEventListener("change", () => {
      if (radio.value === "fast") {
        modeFastCard.classList.add("selected");
        modeAccurateCard.classList.remove("selected");
      } else {
        modeAccurateCard.classList.add("selected");
        modeFastCard.classList.remove("selected");
      }
    });
  });

  function parseGradioResult(resultData) {
    const [, top3Obj, tableObj, galleryArr, infoText] = resultData || [];

    const exactPctMap = {};
    if (top3Obj && Array.isArray(top3Obj.confidences)) {
      top3Obj.confidences.forEach((item) => {
        if (item && item.label != null && typeof item.confidence === "number") {
          exactPctMap[item.label] = item.confidence * 100.0;
        }
      });
    }

    const rows = tableObj && Array.isArray(tableObj.data) ? tableObj.data : [];
    const predictions = rows.map((row, idx) => {
      const rank = row[0] ?? idx + 1;
      const name = String(row[1] ?? "");
      const rawPct = parseFloat(String(row[2] ?? "0").replace("%", "").trim()) || 0;
      const pct = exactPctMap[name] !== undefined ? exactPctMap[name] : rawPct;
      return { rank, name, pct };
    });

    let elapsed_s = "—";
    let duration_s = "—";
    if (typeof infoText === "string") {
      const m = infoText.match(/inferensi:\s*([\d.]+).*?Durasi video:\s*([\d.]+)/i);
      if (m) {
        elapsed_s = m[1];
        duration_s = m[2];
      }
    }

    const frameUrls = Array.isArray(galleryArr)
      ? galleryArr
          .map((item) => item?.image?.url || item?.url || (typeof item === "string" ? item : null))
          .filter(Boolean)
      : [];

    return {
      predictions,
      elapsed_s,
      duration_s,
      frameUrls,
    };
  }

  classifyBtn.addEventListener("click", async () => {
    showError("");
    if (!selectedFile && !usingSampleVideo) {
      showError("Please upload a video clip first or click 'Load Sample Clip (contoh.mp4)'.");
      return;
    }

    const mode = document.querySelector('input[name="mode"]:checked').value;
    const modeLabel = GRADIO_MODE_LABELS[mode] || GRADIO_MODE_LABELS.fast;

    classifyBtn.disabled = true;
    classifyBtnText.textContent = "Connecting to Hugging Face Space...";
    metaBadge.textContent = "Connecting to HF Space...";

    try {
      let videoBlob = selectedFile;
      if (usingSampleVideo) {
        const sampleRes = await fetch(SAMPLE_VIDEO_URL);
        if (!sampleRes.ok) {
          throw new Error("Unable to load sample video (contoh.mp4).");
        }
        const blob = await sampleRes.blob();
        videoBlob = new File([blob], "contoh.mp4", { type: "video/mp4" });
      }

      const client = await getGradioClient();
      classifyBtnText.textContent = "Analyzing 16 Frames on HF Space...";
      metaBadge.textContent = "Running ONNX inference on HF Space...";

      const result = await client.predict("/classify", [
        handle_file(videoBlob),
        modeLabel,
      ]);

      const parsed = parseGradioResult(result.data);
      if (!parsed.predictions || parsed.predictions.length === 0) {
        throw new Error("Received an empty prediction response from Hugging Face Space.");
      }

      renderResults(parsed);
    } catch (err) {
      const msg = err?.message || String(err) || "An error occurred while communicating with Hugging Face Space.";
      showError(msg);
      metaBadge.textContent = "Processing failed";
    } finally {
      classifyBtn.disabled = false;
      classifyBtnText.textContent = "Classify Skating Element";
    }
  });

  function renderResults(data) {
    emptyState.classList.add("hidden");
    resultsContent.classList.remove("hidden");

    const top = data.predictions[0];
    topClassName.textContent = top.name;
    topClassCategory.textContent = getElementCategory(top.name);
    topConfidence.textContent = `${top.pct.toFixed(1)}%`;

    metaBadge.textContent = `Inference: ${data.elapsed_s}s • Duration: ${data.duration_s}s`;

    top3Bars.innerHTML = "";
    data.predictions.slice(0, 3).forEach((item, idx) => {
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <div class="prob-row-top">
          <span>#${idx + 1} ${item.name}</span>
          <span>${item.pct.toFixed(1)}%</span>
        </div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: ${Math.max(item.pct, 1.5)}%"></div>
        </div>
      `;
      top3Bars.appendChild(row);
    });

    allClassesBody.innerHTML = "";
    data.predictions.forEach((item, idx) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${idx + 1}</td>
        <td><strong>${item.name}</strong></td>
        <td>${getElementCategory(item.name)}</td>
        <td>${item.pct.toFixed(1)}%</td>
      `;
      allClassesBody.appendChild(tr);
    });

    if (data.frameUrls && data.frameUrls.length > 0) {
      framesSection.classList.remove("hidden");
      framesGallery.innerHTML = "";
      data.frameUrls.forEach((url, i) => {
        const div = document.createElement("div");
        div.className = "frame-item";
        div.innerHTML = `
          <img src="${url}" alt="Sampled frame ${i + 1} of 16" loading="lazy" />
          <span class="frame-idx">F${i + 1}</span>
        `;
        framesGallery.appendChild(div);
      });
    } else if (data.frames_b64 && data.frames_b64.length > 0) {
      framesSection.classList.remove("hidden");
      framesGallery.innerHTML = "";
      data.frames_b64.forEach((b64, i) => {
        const div = document.createElement("div");
        div.className = "frame-item";
        div.innerHTML = `
          <img src="data:image/jpeg;base64,${b64}" alt="Sampled frame ${i + 1} of 16" loading="lazy" />
          <span class="frame-idx">F${i + 1}</span>
        `;
        framesGallery.appendChild(div);
      });
    }
  }

  // =========================================================================
  // 3. SECTION 03: SKATER GALLERY (SkaterStrip, SkaterTile, SkaterDetail)
  // =========================================================================
  const skaterSearchInput = document.getElementById("skaterSearchInput");
  const disciplineFilter = document.getElementById("disciplineFilter");
  const countryFilter = document.getElementById("countryFilter");
  const resetSkaterFiltersBtn = document.getElementById("resetSkaterFiltersBtn");

  const skaterStripCount = document.getElementById("skaterStripCount");
  const stripPrevBtn = document.getElementById("stripPrevBtn");
  const stripNextBtn = document.getElementById("stripNextBtn");
  const skaterStripStage = document.querySelector(".skater-strip-stage");
  const skaterStripTrack = document.getElementById("skaterStripTrack");
  const skaterStripEmpty = document.getElementById("skaterStripEmpty");

  const skaterDetailPanel = document.getElementById("skaterDetailPanel");
  const detailCountryBadge = document.getElementById("detailCountryBadge");
  const detailDisciplineBadge = document.getElementById("detailDisciplineBadge");
  const detailGivenName = document.getElementById("detailGivenName");
  const detailSurname = document.getElementById("detailSurname");
  const detailNicknameWrap = document.getElementById("detailNicknameWrap");
  const detailNicknameBadge = document.getElementById("detailNicknameBadge");
  const detailNicknameReason = document.getElementById("detailNicknameReason");
  const detailBioText = document.getElementById("detailBioText");
  const detailAchievementsList = document.getElementById("detailAchievementsList");
  const detailSourcesWrap = document.getElementById("detailSourcesWrap");
  const detailSourcesList = document.getElementById("detailSourcesList");
  const detailVideoTabs = document.getElementById("detailVideoTabs");
  const detailHighlightsContainer = document.getElementById("detailHighlightsContainer");

  const YT_ID_REGEX = /^[A-Za-z0-9_-]{11}$/;
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let allSkaters = [];
  let filteredSkaters = [];
  let selectedSkaterId = null;
  let detailFadeTimer = null;

  function parseSkaterNameParts(skater) {
    const raw = (skater.name || "").trim();
    if (raw.includes("/")) {
      const partners = raw.split("/").map((p) => p.trim()).filter(Boolean);
      const surnames = partners.map((p) => {
        const tokens = p.split(/\s+/);
        return tokens[tokens.length - 1] || p;
      });
      const givens = partners.map((p) => {
        const tokens = p.split(/\s+/);
        return tokens.length > 1 ? tokens.slice(0, -1).join(" ") : p;
      });
      return {
        isPair: true,
        tileSurnamesHtml: surnames.map(escapeHtml).join("<br>"),
        tileSubText: `${givens.join(" / ")} • ${skater.countryCode}`,
        headingDark: `${partners[0] || ""}`,
        headingBlue: partners[1] ? `/ ${partners[1]}` : "",
      };
    }

    const parts = raw.split(/\s+/).filter(Boolean);
    const surname = parts.length > 1 ? parts[parts.length - 1] : raw;
    const given = parts.length > 1 ? parts.slice(0, -1).join(" ") : "";

    return {
      isPair: false,
      tileSurnamesHtml: escapeHtml(surname),
      tileSubText: given ? `${given} • ${skater.countryCode}` : skater.countryCode,
      headingDark: given || surname,
      headingBlue: given ? surname : "",
    };
  }

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function buildCountryFlagSvg(countryCode, countryName) {
    const code = String(countryCode || "").toUpperCase();
    const label = escapeHtml(countryName || code);
    let inner = "";
    if (code === "JPN") {
      inner = `
        <rect width="20" height="14" fill="#FFFFFF" />
        <circle cx="10" cy="7" r="4" fill="#BC002D" />
      `;
    } else if (code === "RUS") {
      inner = `
        <rect width="20" height="4.67" y="0" fill="#FFFFFF" />
        <rect width="20" height="4.67" y="4.67" fill="#0039A6" />
        <rect width="20" height="4.67" y="9.33" fill="#D52B1E" />
      `;
    } else if (code === "USA") {
      inner = `
        <rect width="20" height="14" fill="#FFFFFF" />
        <rect width="20" height="2" y="0" fill="#B22234" />
        <rect width="20" height="2" y="4" fill="#B22234" />
        <rect width="20" height="2" y="8" fill="#B22234" />
        <rect width="20" height="2" y="12" fill="#B22234" />
        <rect width="9" height="8" x="0" y="0" fill="#3C3B6E" />
        <circle cx="2.5" cy="2.2" r="0.6" fill="#FFFFFF" />
        <circle cx="4.5" cy="2.2" r="0.6" fill="#FFFFFF" />
        <circle cx="6.5" cy="2.2" r="0.6" fill="#FFFFFF" />
        <circle cx="3.5" cy="4" r="0.6" fill="#FFFFFF" />
        <circle cx="5.5" cy="4" r="0.6" fill="#FFFFFF" />
        <circle cx="2.5" cy="5.8" r="0.6" fill="#FFFFFF" />
        <circle cx="4.5" cy="5.8" r="0.6" fill="#FFFFFF" />
        <circle cx="6.5" cy="5.8" r="0.6" fill="#FFFFFF" />
      `;
    } else if (code === "KOR") {
      inner = `
        <rect width="20" height="14" fill="#FFFFFF" />
        <circle cx="10" cy="7" r="3.4" fill="#CD2E3A" />
        <path d="M 6.6 7 A 3.4 3.4 0 0 0 13.4 7 A 1.7 1.7 0 0 1 10 7 A 1.7 1.7 0 0 0 6.6 7 Z" fill="#0047A0" />
        <line x1="3.4" y1="3.2" x2="5.4" y2="1.8" stroke="#0F172A" stroke-width="1" />
        <line x1="14.6" y1="1.8" x2="16.6" y2="3.2" stroke="#0F172A" stroke-width="1" />
        <line x1="3.4" y1="10.8" x2="5.4" y2="12.2" stroke="#0F172A" stroke-width="1" />
        <line x1="14.6" y1="12.2" x2="16.6" y2="10.8" stroke="#0F172A" stroke-width="1" />
      `;
    } else {
      // AIN / Neutral Athlete Flag
      inner = `
        <rect width="20" height="14" fill="#0F766E" />
        <circle cx="10" cy="7" r="3.6" fill="none" stroke="#FFFFFF" stroke-width="1.2" />
        <circle cx="10" cy="7" r="1.2" fill="#FFFFFF" />
      `;
    }
    return `
      <svg class="flag-svg" viewBox="0 0 20 14" width="20" height="14" role="img" aria-label="${label} flag">
        ${inner}
        <rect x="0.25" y="0.25" width="19.5" height="13.5" fill="none" stroke="rgba(2,8,23,0.18)" stroke-width="0.5" />
      </svg>
    `;
  }

  function buildNeutralSilhouetteSvg(skaterName) {
    return `
      <svg
        class="skater-tile-silhouette"
        viewBox="0 0 120 150"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Neutral portrait silhouette for ${escapeHtml(skaterName)}"
      >
        <circle cx="60" cy="46" r="24" fill="currentColor" fill-opacity="0.38" />
        <path
          d="M20 142 C20 104, 38 86, 60 86 C82 86, 100 104, 100 142 Z"
          fill="currentColor"
          fill-opacity="0.38"
        />
      </svg>
    `;
  }

  // COMPONENT: SkaterTile
  function createSkaterTile(skater, isSelected, onSelect) {
    const nameParts = parseSkaterNameParts(skater);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "skater-tile";
    btn.dataset.skaterId = skater.id;
    btn.setAttribute("aria-pressed", isSelected ? "true" : "false");
    btn.setAttribute("aria-label", `${skater.name}, ${skater.country}, ${skater.discipline}`);

    const hasPhoto = typeof skater.photo === "string" && skater.photo.trim().length > 0;
    const mediaHtml = hasPhoto
      ? `<img class="skater-tile-photo" src="${escapeHtml(skater.photo.trim())}" alt="Portrait of ${escapeHtml(skater.name)}" loading="lazy" decoding="async" />`
      : buildNeutralSilhouetteSvg(skater.name);

    const nicknameStr = typeof skater.nickname === "string" ? skater.nickname.trim() : "";
    const nicknameHtml =
      nicknameStr && !nameParts.isPair
        ? `<span class="tile-nickname">${escapeHtml(nicknameStr)}</span>`
        : "";

    const flagSvg = buildCountryFlagSvg(skater.countryCode, skater.country);

    btn.innerHTML = `
      <div class="skater-tile-surface">
        <div class="skater-tile-media">
          ${mediaHtml}
        </div>
        <div class="skater-tile-namebar">
          <span class="tile-surname">${nameParts.tileSurnamesHtml}</span>
          <span class="tile-given-country">${flagSvg}<span>${escapeHtml(nameParts.tileSubText)}</span></span>
          ${nicknameHtml}
        </div>
      </div>
      <svg class="skater-tile-border" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        <polygon points="16,0 100,0 84,100 0,100" />
      </svg>
    `;

    if (hasPhoto) {
      const imgEl = btn.querySelector(".skater-tile-photo");
      if (imgEl) {
        imgEl.addEventListener("error", () => {
          const mediaWrap = btn.querySelector(".skater-tile-media");
          if (mediaWrap) {
            mediaWrap.innerHTML = buildNeutralSilhouetteSvg(skater.name);
          }
        });
      }
    }

    btn.addEventListener("click", () => onSelect(skater.id, { focusTile: false }));
    return btn;
  }

  function extractValidYoutubeItems(highlights) {
    if (!Array.isArray(highlights)) return [];
    const valid = [];
    highlights.forEach((item, idx) => {
      if (typeof item === "string") {
        const id = item.trim();
        if (YT_ID_REGEX.test(id)) {
          valid.push({ id, title: `Video ${idx + 1}` });
        }
      } else if (item && typeof item === "object") {
        const rawId = String(item.youtubeId || item.id || "").trim();
        if (YT_ID_REGEX.test(rawId)) {
          valid.push({
            id: rawId,
            title: item.title ? String(item.title) : `Video ${idx + 1}`,
          });
        }
      }
    });
    return valid;
  }

  function renderHighlightVideoEmbed(skater, validVideos, activeVideoIdx) {
    detailVideoTabs.innerHTML = "";
    detailHighlightsContainer.innerHTML = "";

    if (validVideos.length === 0) {
      const emptyVideo = document.createElement("div");
      emptyVideo.className = "detail-placeholder-box";
      emptyVideo.textContent =
        "No valid 11-character YouTube ID set yet. Add YouTube IDs to the \"highlights\" array in static/skaters.json.";
      detailHighlightsContainer.appendChild(emptyVideo);
      return;
    }

    if (validVideos.length > 1) {
      validVideos.forEach((vid, idx) => {
        const tabBtn = document.createElement("button");
        tabBtn.type = "button";
        tabBtn.className = `btn ${idx === activeVideoIdx ? "btn-primary" : "btn-secondary"} btn-sm`;
        tabBtn.setAttribute("role", "tab");
        tabBtn.setAttribute("aria-selected", idx === activeVideoIdx ? "true" : "false");
        tabBtn.textContent = `Video ${idx + 1}`;
        tabBtn.addEventListener("click", () => {
          renderHighlightVideoEmbed(skater, validVideos, idx);
        });
        detailVideoTabs.appendChild(tabBtn);
      });
    }

    const chosen = validVideos[activeVideoIdx] || validVideos[0];
    const embedUrl = `https://www.youtube-nocookie.com/embed/${chosen.id}?rel=0`;
    const watchUrl = `https://www.youtube.com/watch?v=${chosen.id}`;
    const wrap = document.createElement("div");
    wrap.className = "highlight-embed-card";
    wrap.innerHTML = `
      <div class="iframe-aspect">
        <iframe
          src="${embedUrl}"
          title="${escapeHtml(chosen.title)} — ${escapeHtml(skater.name)}"
          loading="lazy"
          referrerpolicy="strict-origin-when-cross-origin"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          allowfullscreen
        ></iframe>
      </div>
      <div class="highlight-caption">
        <strong>${escapeHtml(chosen.title)}</strong>
        <a href="${watchUrl}" target="_blank" rel="noopener noreferrer">Watch on YouTube (${escapeHtml(chosen.id)})</a>
      </div>
    `;
    detailHighlightsContainer.appendChild(wrap);
  }

  // COMPONENT: SkaterDetail
  function updateSkaterDetailPanel(skater, animate = true) {
    if (!skater) {
      skaterDetailPanel.classList.add("hidden");
      detailHighlightsContainer.innerHTML = "";
      return;
    }

    skaterDetailPanel.classList.remove("hidden");

    const applyContent = () => {
      const nameParts = parseSkaterNameParts(skater);
      detailCountryBadge.innerHTML = `${buildCountryFlagSvg(skater.countryCode, skater.country)}<span>${escapeHtml(skater.countryCode)} • ${escapeHtml(skater.country)}</span>`;
      detailDisciplineBadge.textContent = skater.discipline;
      detailGivenName.textContent = nameParts.headingDark;
      detailSurname.textContent = nameParts.headingBlue;

      const nickStr = typeof skater.nickname === "string" ? skater.nickname.trim() : "";
      const nickReasonStr = typeof skater.nicknameReason === "string" ? skater.nicknameReason.trim() : "";
      if (nickStr) {
        detailNicknameWrap.classList.remove("hidden");
        detailNicknameBadge.textContent = nickStr;
        if (nickReasonStr) {
          detailNicknameReason.classList.remove("hidden");
          detailNicknameReason.textContent = nickReasonStr;
        } else {
          detailNicknameReason.classList.add("hidden");
          detailNicknameReason.textContent = "";
        }
      } else {
        detailNicknameWrap.classList.add("hidden");
        detailNicknameBadge.textContent = "";
        detailNicknameReason.classList.add("hidden");
        detailNicknameReason.textContent = "";
      }

      const bioStr = (skater.bio || "").trim();
      if (bioStr) {
        detailBioText.className = "detail-bio-text";
        detailBioText.textContent = bioStr;
      } else {
        detailBioText.className = "detail-placeholder-box";
        detailBioText.textContent =
          "Bio placeholder — fill in the \"bio\" field for this skater in static/skaters.json.";
      }

      detailAchievementsList.innerHTML = "";
      const achs = Array.isArray(skater.achievements) ? skater.achievements.filter(Boolean) : [];
      if (achs.length > 0) {
        achs.forEach((ach) => {
          const li = document.createElement("li");
          li.textContent = ach;
          detailAchievementsList.appendChild(li);
        });
      } else {
        const li = document.createElement("li");
        li.textContent =
          "Achievements placeholder — add entries to the \"achievements\" array in static/skaters.json.";
        detailAchievementsList.appendChild(li);
      }

      detailSourcesList.innerHTML = "";
      const srcs = Array.isArray(skater.sources)
        ? skater.sources.filter((u) => typeof u === "string" && /^https?:\/\//i.test(u.trim()))
        : [];
      if (srcs.length > 0) {
        detailSourcesWrap.classList.remove("hidden");
        srcs.forEach((url, idx) => {
          const li = document.createElement("li");
          let host = "Source";
          try {
            host = new URL(url.trim()).hostname.replace(/^www\./, "");
          } catch (_) {}
          const a = document.createElement("a");
          a.href = url.trim();
          a.target = "_blank";
          a.rel = "noopener noreferrer";
          a.textContent = `Source ${idx + 1} (${host})`;
          li.appendChild(a);
          detailSourcesList.appendChild(li);
        });
      } else {
        detailSourcesWrap.classList.add("hidden");
      }

      const validVideos = extractValidYoutubeItems(skater.highlights);
      renderHighlightVideoEmbed(skater, validVideos, 0);
    };

    if (!animate || prefersReducedMotion) {
      applyContent();
      skaterDetailPanel.classList.remove("is-fading");
      return;
    }

    if (detailFadeTimer) clearTimeout(detailFadeTimer);
    skaterDetailPanel.classList.add("is-fading");
    detailFadeTimer = setTimeout(() => {
      applyContent();
      skaterDetailPanel.classList.remove("is-fading");
    }, 150);
  }

  function selectSkaterById(skaterId, options = { focusTile: false, scrollIntoView: true, animateDetail: true }) {
    const target = filteredSkaters.find((s) => s.id === skaterId);
    if (!target) return;

    const changed = selectedSkaterId !== target.id;
    selectedSkaterId = target.id;

    const tiles = skaterStripTrack.querySelectorAll(".skater-tile");
    let activeTileEl = null;
    tiles.forEach((tile) => {
      const isMatch = tile.dataset.skaterId === selectedSkaterId;
      tile.setAttribute("aria-pressed", isMatch ? "true" : "false");
      if (isMatch) activeTileEl = tile;
    });

    if (activeTileEl) {
      if (options.focusTile) {
        activeTileEl.focus({ preventScroll: true });
      }
      if (options.scrollIntoView !== false) {
        activeTileEl.scrollIntoView({
          behavior: prefersReducedMotion ? "auto" : "smooth",
          block: "nearest",
          inline: "center",
        });
      }
    }

    updateSkaterDetailPanel(target, changed && options.animateDetail !== false);
  }

  function stepSkaterSelection(delta, focusTile = false) {
    if (filteredSkaters.length === 0) return;
    const curIdx = filteredSkaters.findIndex((s) => s.id === selectedSkaterId);
    const nextIdx = curIdx === -1 ? 0 : Math.max(0, Math.min(filteredSkaters.length - 1, curIdx + delta));
    selectSkaterById(filteredSkaters[nextIdx].id, {
      focusTile,
      scrollIntoView: true,
      animateDetail: true,
    });
  }

  // COMPONENT: SkaterStrip
  function renderSkaterStrip() {
    const query = (skaterSearchInput.value || "").trim().toLowerCase();
    const disc = disciplineFilter.value;
    const country = countryFilter.value;

    filteredSkaters = allSkaters.filter((s) => {
      const matchName =
        !query ||
        s.name.toLowerCase().includes(query) ||
        s.country.toLowerCase().includes(query) ||
        s.countryCode.toLowerCase().includes(query);
      const matchDisc = disc === "ALL" || s.discipline === disc;
      const matchCountry = country === "ALL" || s.countryCode === country;
      return matchName && matchDisc && matchCountry;
    });

    skaterStripCount.textContent = `${filteredSkaters.length} ${filteredSkaters.length === 1 ? "Skater" : "Skaters"}`;
    skaterStripTrack.innerHTML = "";

    if (filteredSkaters.length === 0) {
      skaterStripStage.classList.add("hidden");
      skaterStripEmpty.classList.remove("hidden");
      stripPrevBtn.disabled = true;
      stripNextBtn.disabled = true;
      updateSkaterDetailPanel(null, false);
      return;
    }

    skaterStripStage.classList.remove("hidden");
    skaterStripEmpty.classList.add("hidden");
    stripPrevBtn.disabled = filteredSkaters.length <= 1;
    stripNextBtn.disabled = filteredSkaters.length <= 1;

    // Default to first skater if current selection is not in filtered results
    if (!filteredSkaters.some((s) => s.id === selectedSkaterId)) {
      selectedSkaterId = filteredSkaters[0].id;
    }

    filteredSkaters.forEach((skater) => {
      const isSelected = skater.id === selectedSkaterId;
      const tileEl = createSkaterTile(skater, isSelected, (id, opts) => {
        selectSkaterById(id, {
          focusTile: opts.focusTile,
          scrollIntoView: true,
          animateDetail: true,
        });
      });
      skaterStripTrack.appendChild(tileEl);
    });

    const activeSkater = filteredSkaters.find((s) => s.id === selectedSkaterId) || filteredSkaters[0];
    updateSkaterDetailPanel(activeSkater, false);
  }

  stripPrevBtn.addEventListener("click", () => stepSkaterSelection(-1, false));
  stripNextBtn.addEventListener("click", () => stepSkaterSelection(1, false));

  skaterStripTrack.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") {
      e.preventDefault();
      stepSkaterSelection(1, true);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      stepSkaterSelection(-1, true);
    }
  });

  async function initSkatersDirectory() {
    try {
      const res = await fetch(SKATERS_DATA_URL, { cache: "no-store" });
      if (!res.ok) throw new Error("Failed to load skaters.json");
      allSkaters = await res.json();

      const countries = Array.from(new Set(allSkaters.map((s) => s.countryCode))).sort();
      countries.forEach((code) => {
        const sample = allSkaters.find((s) => s.countryCode === code);
        const opt = document.createElement("option");
        opt.value = code;
        opt.textContent = `${sample ? sample.country : code} (${code})`;
        countryFilter.appendChild(opt);
      });

      if (allSkaters.length > 0) {
        selectedSkaterId = allSkaters[0].id;
      }

      renderSkaterStrip();
    } catch (err) {
      skaterStripStage.classList.add("hidden");
      skaterStripEmpty.classList.remove("hidden");
      skaterStripEmpty.textContent = "Unable to load static/skaters.json.";
    }
  }

  skaterSearchInput.addEventListener("input", renderSkaterStrip);
  disciplineFilter.addEventListener("change", renderSkaterStrip);
  countryFilter.addEventListener("change", renderSkaterStrip);
  resetSkaterFiltersBtn.addEventListener("click", () => {
    skaterSearchInput.value = "";
    disciplineFilter.value = "ALL";
    countryFilter.value = "ALL";
    renderSkaterStrip();
  });

  initSkatersDirectory();

  // =========================================================================
  // 4. SECTION 05: PROJECT TEAM (from static/team.json)
  // =========================================================================
  const teamGrid = document.getElementById("teamGrid");

  async function initTeamSection() {
    if (!teamGrid) return;
    try {
      const res = await fetch("/static/team.json", { cache: "no-store" });
      if (!res.ok) throw new Error("Failed to load team.json");
      const members = await res.json();

      teamGrid.innerHTML = "";
      members.forEach((m) => {
        const card = document.createElement("article");
        card.className = "ui-card team-card";

        const avatarHtml =
          m.photo && m.photo.trim()
            ? `<img src="${escapeHtml(m.photo.trim())}" alt="Photo of ${escapeHtml(m.name)}" loading="lazy" decoding="async" />`
            : `<span>${escapeHtml(m.initials || m.name.slice(0, 2).toUpperCase())}</span>`;

        const linkedinUrl = typeof m.linkedin === "string" ? m.linkedin.trim() : "";
        const hasValidLinkedin = /^https?:\/\//i.test(linkedinUrl);

        card.innerHTML = `
          <div class="team-avatar" aria-hidden="true">${avatarHtml}</div>
          <div class="team-card-body">
            <div>
              <span class="team-role-prominent">${escapeHtml(m.role)}</span>
              <h3 class="team-name">${escapeHtml(m.name)}</h3>
              <div class="team-campus">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <path d="M22 10v6M2 10l10-5 10 5-10 5z"/>
                  <path d="M6 12v5c0 2 3 3 6 3s6-1 6-3v-5"/>
                </svg>
                <span>${escapeHtml(m.campus)}</span>
              </div>
            </div>
            <div class="team-card-footer">
              ${
                hasValidLinkedin
                  ? `<a href="${escapeHtml(linkedinUrl)}" target="_blank" rel="noopener noreferrer" class="btn btn-primary btn-sm team-linkedin-btn" aria-label="LinkedIn profile of ${escapeHtml(m.name)}">
                      <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                        <path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.28 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.75M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z"/>
                      </svg>
                      <span>LinkedIn</span>
                    </a>`
                  : `<span class="badge-mono">LinkedIn not set</span>`
              }
            </div>
          </div>
        `;
        teamGrid.appendChild(card);
      });
    } catch (err) {
      teamGrid.innerHTML = `<div class="ui-card" style="grid-column:1/-1;">Unable to load static/team.json.</div>`;
    }
  }

  function initMobileNav() {
    const burgerBtn = document.getElementById("navBurgerBtn");
    const mobileMenu = document.getElementById("mobileNavMenu");
    if (!burgerBtn || !mobileMenu) return;

    function closeMenu() {
      burgerBtn.classList.remove("is-open");
      burgerBtn.setAttribute("aria-expanded", "false");
      mobileMenu.classList.remove("is-open");
      window.setTimeout(() => {
        if (!burgerBtn.classList.contains("is-open")) {
          mobileMenu.hidden = true;
        }
      }, 180);
    }

    function openMenu() {
      mobileMenu.hidden = false;
      requestAnimationFrame(() => {
        burgerBtn.classList.add("is-open");
        burgerBtn.setAttribute("aria-expanded", "true");
        mobileMenu.classList.add("is-open");
      });
    }

    burgerBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (burgerBtn.classList.contains("is-open")) {
        closeMenu();
      } else {
        openMenu();
      }
    });

    mobileMenu.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", closeMenu);
    });

    document.addEventListener("click", (e) => {
      if (!burgerBtn.classList.contains("is-open")) return;
      if (!mobileMenu.contains(e.target) && !burgerBtn.contains(e.target)) {
        closeMenu();
      }
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && burgerBtn.classList.contains("is-open")) {
        closeMenu();
      }
    });

    window.addEventListener("resize", () => {
      if (window.innerWidth > 900 && burgerBtn.classList.contains("is-open")) {
        closeMenu();
      }
    });
  }

  initMobileNav();
  initTeamSection();
});
