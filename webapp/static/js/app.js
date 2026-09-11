/**
 * RetinaGuard Web Application Controller
 * Professional clinical screening workstation logic.
 */

document.addEventListener("DOMContentLoaded", () => {
  // State
  let currentFile = null;
  let currentDemoId = null;
  let currentAnalysis = null;
  let currentOverlays = {};
  let demoCases = [];
  let currentRequestId = 0;

  // DOM Elements
  const selectDemoCase = document.getElementById("select-demo-case");
  const btnLoadDemo = document.getElementById("btn-load-demo");
  const btnTriggerUpload = document.getElementById("btn-trigger-upload");
  const fileUpload = document.getElementById("file-upload");
  const btnAnalyze = document.getElementById("btn-analyze");
  const analyzeSpinner = document.getElementById("analyze-spinner");
  const analyzeText = document.getElementById("analyze-text");
  
  // Patient fields
  const inputPatientId = document.getElementById("input-patient-id");
  const inputExamId = document.getElementById("input-exam-id");
  const selectEye = document.getElementById("select-eye");

  // Viewport
  const emptyPlaceholder = document.getElementById("empty-placeholder");
  const mainImageView = document.getElementById("main-image-view");
  const processingOverlay = document.getElementById("processing-overlay");
  const imageMetaTag = document.getElementById("image-meta-tag");
  const layerToggles = document.getElementById("layer-toggles");
  const layerLegend = document.getElementById("layer-legend");
  const safetyAlertBox = document.getElementById("safety-alert-box");
  const borderlineAlertBox = document.getElementById("borderline-alert-box");

  // Quality Panel
  const badgeQualityStatus = document.getElementById("badge-quality-status");
  const valQualityScore = document.getElementById("val-quality-score");
  const qualityMeterFill = document.getElementById("quality-meter-fill");
  const valQFocus = document.getElementById("val-q-focus");
  const valQIllum = document.getElementById("val-q-illum");
  const valQContrast = document.getElementById("val-q-contrast");
  const valQFov = document.getElementById("val-q-fov");
  const boxQualityReason = document.getElementById("box-quality-reason");
  const textQualityReason = document.getElementById("text-quality-reason");

  // DR Panel
  const badgeReferralStatus = document.getElementById("badge-referral-status");
  const textGradeHeadline = document.getElementById("text-grade-headline");
  const textGradeConfidence = document.getElementById("text-grade-confidence");
  const demoComparisonBox = document.getElementById("demo-comparison-box");
  const valReferenceGrade = document.getElementById("val-reference-grade");
  const valAiPrediction = document.getElementById("val-ai-prediction");
  const probabilitiesSection = document.getElementById("probabilities-section");
  const probBarsContainer = document.getElementById("prob-bars-container");

  // Table & Lesions
  const badgeLesionMode = document.getElementById("badge-lesion-mode");
  const valOdStatus = document.getElementById("val-od-status");
  const valOdDetails = document.getElementById("val-od-details");
  const valFoveaStatus = document.getElementById("val-fovea-status");
  const valFoveaDetails = document.getElementById("val-fovea-details");
  const valVesselDensity = document.getElementById("val-vessel-density");
  const valMaCount = document.getElementById("val-ma-count");
  const valMaNotes = document.getElementById("val-ma-notes");
  const valExCount = document.getElementById("val-ex-count");
  const valExNotes = document.getElementById("val-ex-notes");
  const valHeCount = document.getElementById("val-he-count");
  const valHeNotes = document.getElementById("val-he-notes");
  const valSeCount = document.getElementById("val-se-count");
  const valSeNotes = document.getElementById("val-se-notes");
  const valNvRisk = document.getElementById("val-nv-risk");

  // Recommendation & Report
  const recTitle = document.getElementById("rec-title");
  const recBody = document.getElementById("rec-body");
  const reportActionBar = document.getElementById("report-action-bar");
  const btnViewReport = document.getElementById("btn-view-report");
  const btnDownloadReport = document.getElementById("btn-download-report");

  // About modal
  const btnAbout = document.getElementById("btn-about");
  const aboutModal = document.getElementById("about-modal");
  const btnCloseAbout = document.getElementById("btn-close-about");

  // 1. Fetch and populate demo cases
  async function loadDemoCases() {
    try {
      const resp = await fetch("/api/demo-cases");
      if (!resp.ok) return;
      demoCases = await resp.json();
      if (!demoCases || demoCases.length === 0) {
        selectDemoCase.innerHTML = '<option value="">-- No demo scans bundled (Upload below) --</option>';
        selectDemoCase.disabled = true;
        btnLoadDemo.disabled = true;
      } else {
        selectDemoCase.innerHTML = '<option value="">Choose Demo Grade</option>';
        demoCases.forEach(c => {
          const opt = document.createElement("option");
          opt.value = c.id;
          opt.textContent = c.label;
          selectDemoCase.appendChild(opt);
        });
        selectDemoCase.disabled = false;
        btnLoadDemo.disabled = false;
      }
    } catch (err) {
      console.error("Failed to load demo cases:", err);
    }
  }
  loadDemoCases();

  // 2. Demo Case Selection Handlers
  function handleSelectDemoCase(demoId) {
    if (!demoId) {
      clearSelection();
      return;
    }

    const selectedCase = demoCases.find(c => c.id === demoId);
    if (!selectedCase) return;

    // Associate current active state
    currentDemoId = demoId;
    currentFile = null;
    fileUpload.value = "";
    selectDemoCase.value = demoId;
    currentRequestId++;

    showImage(selectedCase.image_url, selectedCase.label);
    imageMetaTag.textContent = `${selectedCase.label}`;
    imageMetaTag.className = "badge badge-neutral";

    // Immediately wipe out all previous analysis results so UI never shows stale data
    resetResults(`Loaded authentic reference case (${selectedCase.reference_name}). Click Analyze Image ▶.`);
    btnAnalyze.disabled = false;
  }

  function clearSelection() {
    currentDemoId = null;
    currentFile = null;
    fileUpload.value = "";
    selectDemoCase.value = "";
    currentRequestId++;

    emptyPlaceholder.style.display = "block";
    mainImageView.style.display = "none";
    mainImageView.src = "";
    imageMetaTag.textContent = "No Image Loaded";
    imageMetaTag.className = "badge badge-neutral";

    resetResults("Select a demo case or upload an image to begin screening.");
    btnAnalyze.disabled = true;
  }

  // Changing dropdown immediately switches case and clears stale results
  selectDemoCase.addEventListener("change", (e) => {
    handleSelectDemoCase(e.target.value);
  });

  // Load Demo button also loads the chosen case
  btnLoadDemo.addEventListener("click", () => {
    handleSelectDemoCase(selectDemoCase.value);
  });

  // 3. Upload handling
  btnTriggerUpload.addEventListener("click", () => fileUpload.click());

  fileUpload.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    currentFile = file;
    currentDemoId = null;
    selectDemoCase.value = "";
    currentRequestId++;

    // Local Preview
    const objectUrl = URL.createObjectURL(file);
    showImage(objectUrl, file.name);

    // Update metadata badge
    const sizeKb = Math.round(file.size / 1024);
    imageMetaTag.textContent = `${file.name} (${sizeKb} KB)`;
    imageMetaTag.className = "badge badge-neutral";

    resetResults("Image loaded from file. Click Analyze Image ▶ to run screening.");
    btnAnalyze.disabled = false;
  });

  // Viewport image setter
  function showImage(src, alt) {
    emptyPlaceholder.style.display = "none";
    mainImageView.style.display = "block";
    mainImageView.src = src;
    mainImageView.alt = alt;
  }

  // Reset results completely
  function resetResults(msg) {
    // Clear state caches
    currentAnalysis = null;
    currentOverlays = {};

    // Hide overlays & alert banners
    layerToggles.style.display = "none";
    document.querySelectorAll(".toggle-btn").forEach(b => {
      b.classList.toggle("active", b.getAttribute("data-layer") === "original");
    });
    layerLegend.style.display = "none";
    layerLegend.innerHTML = "";
    safetyAlertBox.style.display = "none";
    safetyAlertBox.innerHTML = "";
    borderlineAlertBox.style.display = "none";
    borderlineAlertBox.innerHTML = "";
    probabilitiesSection.style.display = "none";
    probBarsContainer.innerHTML = "";
    reportActionBar.style.display = "none";

    // Quality Panel
    badgeQualityStatus.textContent = "PENDING";
    badgeQualityStatus.className = "badge badge-neutral";
    valQualityScore.textContent = "--";
    qualityMeterFill.style.width = "0%";
    qualityMeterFill.style.background = "#0969da";
    valQFocus.textContent = "--";
    valQIllum.textContent = "--";
    valQContrast.textContent = "--";
    valQFov.textContent = "--";
    boxQualityReason.style.display = "none";
    textQualityReason.textContent = "--";

    // DR Severity Panel
    badgeReferralStatus.textContent = "PENDING";
    badgeReferralStatus.className = "badge badge-neutral";
    textGradeHeadline.textContent = "Awaiting Analysis";
    textGradeHeadline.style.color = "";
    textGradeConfidence.textContent = msg || "Ready to execute screening";
    if (demoComparisonBox) demoComparisonBox.style.display = "none";
    if (valReferenceGrade) valReferenceGrade.textContent = "--";
    if (valAiPrediction) valAiPrediction.textContent = "--";

    // Table & Lesions Panel
    if (badgeLesionMode) {
      badgeLesionMode.textContent = "PENDING";
      badgeLesionMode.className = "badge badge-neutral";
    }
    valOdStatus.textContent = "--";
    if (valOdDetails) valOdDetails.textContent = "Morphological Peak / AI Mask";
    valFoveaStatus.textContent = "--";
    if (valFoveaDetails) valFoveaDetails.textContent = "Temporal Geometric Projection";
    valVesselDensity.textContent = "--";
    valMaCount.textContent = "--";
    if (valMaNotes) valMaNotes.textContent = "Candidate Detection (Top-hat)";
    valExCount.textContent = "--";
    if (valExNotes) valExNotes.textContent = "Candidate Detection (Disc-masked)";
    valHeCount.textContent = "--";
    if (valHeNotes) valHeNotes.textContent = "Candidate Detection (Dark lesions)";
    if (valSeCount) valSeCount.textContent = "--";
    if (valSeNotes) valSeNotes.textContent = "Cotton Wool Spots";
    valNvRisk.textContent = "--";

    // Recommendation & Report
    recTitle.textContent = "Standard Triage Recommendation";
    recBody.textContent = msg || "Upload or load an image to receive clinical guidance.";
    if (btnViewReport) btnViewReport.removeAttribute("href");
    if (btnDownloadReport) btnDownloadReport.removeAttribute("href");
  }

  // 4. Run Analysis
  btnAnalyze.addEventListener("click", async () => {
    if (!currentFile && !currentDemoId) return;

    btnAnalyze.disabled = true;
    analyzeSpinner.style.display = "inline-block";
    analyzeText.textContent = "Analyzing...";
    processingOverlay.style.display = "flex";

    // Stale result guard: tag request with unique ID
    const requestId = ++currentRequestId;

    // Reset results UI before analysis begins so old results never linger
    resetResults("Executing screening pipeline...");

    const formData = new FormData();
    if (currentFile) {
      formData.append("file", currentFile);
    } else if (currentDemoId) {
      formData.append("demo_id", currentDemoId);
    }

    formData.append("patient_id", inputPatientId.value || "PT-82910");
    formData.append("exam_id", inputExamId.value || "EX-00412");
    formData.append("eye", selectEye.value || "Right (OD)");

    try {
      const resp = await fetch("/api/analyze", {
        method: "POST",
        body: formData
      });

      if (!resp.ok) {
        const errorData = await resp.json();
        throw new Error(errorData.detail || "Screening analysis failed.");
      }

      const result = await resp.json();

      // Guard against stale response: if user switched image during analysis, discard!
      if (requestId !== currentRequestId) {
        console.warn("Discarding stale analysis response for older request", requestId);
        return;
      }

      currentAnalysis = result;
      renderAnalysisResults(result);
    } catch (err) {
      if (requestId === currentRequestId) {
        alert("Error: " + err.message);
        resetResults("Analysis error occurred: " + err.message);
      }
    } finally {
      if (requestId === currentRequestId) {
        btnAnalyze.disabled = false;
        analyzeSpinner.style.display = "none";
        analyzeText.textContent = "Analyze Image ▶";
        processingOverlay.style.display = "none";
      }
    }
  });

  // Render Full Results
  function renderAnalysisResults(data) {
    const q = data.quality;

    // 1. Reset any existing alerts
    safetyAlertBox.style.display = "none";
    safetyAlertBox.innerHTML = "";
    borderlineAlertBox.style.display = "none";
    borderlineAlertBox.innerHTML = "";

    // 2. Quality Panel
    valQualityScore.textContent = q.score;
    qualityMeterFill.style.width = `${q.score}%`;
    valQFocus.textContent = q.focus;
    valQIllum.textContent = q.illumination;
    valQContrast.textContent = q.contrast;
    valQFov.textContent = `${q.fov_percent}%`;

    badgeQualityStatus.textContent = q.status;
    if (q.status === "GOOD") {
      badgeQualityStatus.className = "badge badge-good";
      qualityMeterFill.style.background = "#1a7f37";
    } else if (q.status === "BORDERLINE") {
      badgeQualityStatus.className = "badge badge-borderline";
      qualityMeterFill.style.background = "#9a6700";
    } else {
      badgeQualityStatus.className = "badge badge-ungradable";
      qualityMeterFill.style.background = "#cf222e";
    }

    boxQualityReason.style.display = "block";
    textQualityReason.textContent = q.reason;

    // 3. CRITICAL SAFETY GATE HANDLING: UNGRADABLE
    if (data.safety_gate_triggered || !q.gradable) {
      safetyAlertBox.style.display = "block";
      safetyAlertBox.innerHTML = `
        <strong>IMAGE UNGRADABLE:</strong> Downstream grading stopped for patient safety.<br>
        <em>${q.reason}</em><br>
        <strong>Recommended Action:</strong> ${q.recommendation}
      `;

      badgeReferralStatus.textContent = "UNGRADABLE";
      badgeReferralStatus.className = "badge badge-ungradable";
      textGradeHeadline.textContent = "Screening Halted";
      textGradeHeadline.style.color = "#cf222e";

      const ref = data.demo_reference;
      if (ref) {
        textGradeConfidence.innerHTML = `<strong>Reference:</strong> ${ref.label} &bull; <strong>Status:</strong> Rejected by Image Quality Gate (Score: ${q.score}/100)`;
      } else {
        textGradeConfidence.textContent = "Image quality is insufficient to produce a reliable diagnosis.";
      }

      // Hide and clear probabilities for ungradable images
      probabilitiesSection.style.display = "none";
      probBarsContainer.innerHTML = "";

      // Explicitly indicate halted status in Landmarks & Lesions
      if (badgeLesionMode) {
        badgeLesionMode.textContent = "HALTED";
        badgeLesionMode.className = "badge badge-ungradable";
      }
      valOdStatus.textContent = "Halted";
      if (valOdDetails) valOdDetails.textContent = "Downstream analysis withheld";
      valFoveaStatus.textContent = "Halted";
      if (valFoveaDetails) valFoveaDetails.textContent = "Downstream analysis withheld";
      valVesselDensity.textContent = "N/A";
      valMaCount.textContent = "N/A";
      if (valMaNotes) valMaNotes.textContent = "Safety Gate Triggered";
      valExCount.textContent = "N/A";
      if (valExNotes) valExNotes.textContent = "Safety Gate Triggered";
      valHeCount.textContent = "N/A";
      if (valHeNotes) valHeNotes.textContent = "Safety Gate Triggered";
      if (valSeCount) valSeCount.textContent = "N/A";
      if (valSeNotes) valSeNotes.textContent = "Safety Gate Triggered";
      valNvRisk.textContent = "N/A";

      // Recommendation
      recTitle.textContent = "Safety Rejection Guidance";
      recBody.textContent = q.recommendation;

      // Disable layer toggles for ungradable scans
      layerToggles.style.display = "none";
      layerLegend.style.display = "none";
      currentOverlays = {};

      // Enable report download for ungradable record
      if (data.report) {
        setupReportLinks(data.report);
      }
      return;
    }

    // 4. Borderline Alert (Gradable images with borderline quality)
    if (q.status === "BORDERLINE") {
      borderlineAlertBox.style.display = "block";
      borderlineAlertBox.innerHTML = `
        <strong>BORDERLINE QUALITY:</strong> Automated green-channel CLAHE enhancement was applied.<br>
        <em>${q.reason}</em>
      `;
    }

    // 5. Landmarks & Lesions
    const st = data.structures;
    const l = data.lesions;

    valOdStatus.textContent = `Centroid [${st.optic_disc.centroid.join(", ")}]`;
    if (valOdDetails) valOdDetails.textContent = `Radius: ${st.optic_disc.radius_px} px (${st.optic_disc.methodology || "AI/Morphological Mask"})`;
    valFoveaStatus.textContent = `Coords [${st.fovea.coordinates.join(", ")}]`;
    if (valFoveaDetails) valFoveaDetails.textContent = st.fovea.methodology || "Temporal Geometric Projection";
    valVesselDensity.textContent = `${st.vessels.density_percent}%`;

    // Render Mode Badge
    if (badgeLesionMode) {
      if (l.is_ai || l.mode === "AI SEGMENTATION") {
        badgeLesionMode.textContent = "AI SEGMENTATION";
        badgeLesionMode.className = "badge badge-good";
      } else {
        badgeLesionMode.textContent = "HEURISTIC FALLBACK";
        badgeLesionMode.className = "badge badge-borderline";
      }
    }

    if (l.is_ai || l.mode === "AI SEGMENTATION") {
      valMaCount.textContent = `${l.microaneurysms.candidate_count} AI regions`;
      if (valMaNotes) valMaNotes.textContent = "Dual-Head U-Net (IDRiD)";

      valExCount.textContent = `${l.hard_exudates.candidate_count} AI regions (${l.hard_exudates.total_area_px} px)`;
      if (valExNotes) valExNotes.textContent = "Dual-Head U-Net (IDRiD)";

      valHeCount.textContent = `${l.hemorrhages.candidate_count} AI regions (${l.hemorrhages.total_area_px} px)`;
      if (valHeNotes) valHeNotes.textContent = "Dual-Head U-Net (IDRiD)";

      if (valSeCount) {
        valSeCount.textContent = `${l.soft_exudates ? l.soft_exudates.candidate_count : 0} AI regions (${l.soft_exudates ? l.soft_exudates.total_area_px : 0} px)`;
      }
      if (valSeNotes) valSeNotes.textContent = "Dual-Head U-Net (IDRiD)";
    } else {
      valMaCount.textContent = `${l.microaneurysms.candidate_count} candidates`;
      if (valMaNotes) valMaNotes.textContent = "Morphological Top-Hat";

      valExCount.textContent = `${l.hard_exudates.candidate_count} candidates (${l.hard_exudates.total_area_px} px)`;
      if (valExNotes) valExNotes.textContent = "Luminance / Disc-Masked";

      valHeCount.textContent = `${l.hemorrhages.candidate_count} candidates (${l.hemorrhages.total_area_px} px)`;
      if (valHeNotes) valHeNotes.textContent = "Dark Lesion Connected Comp";

      if (valSeCount) valSeCount.textContent = "N/A in fallback";
      if (valSeNotes) valSeNotes.textContent = "Heuristic Fallback";
    }

    valNvRisk.textContent = `${l.neovascularization.risk_level} Risk`;

    // 6. DR Grading (EXP-001)
    const g = data.grading;
    textGradeHeadline.textContent = `Grade ${g.grade} — ${g.severity_name}`;
    textGradeHeadline.style.color = getGradeColor(g.grade);

    const ref = data.demo_reference;
    if (ref && ref.reference_grade !== undefined && ref.reference_grade >= 0) {
      if (demoComparisonBox) demoComparisonBox.style.display = "block";
      if (valReferenceGrade) valReferenceGrade.textContent = `Grade ${ref.reference_grade} — ${ref.reference_name}`;
      if (valAiPrediction) valAiPrediction.textContent = `Grade ${g.grade} — ${g.severity_name}`;
      textGradeConfidence.textContent = `Confidence: ${g.confidence_percent}% (${g.confidence_level} Confidence) • Model: EXP-001`;
    } else {
      if (demoComparisonBox) demoComparisonBox.style.display = "none";
      textGradeConfidence.textContent = `Confidence: ${g.confidence_percent}% (${g.confidence_level} Confidence) • Model: EXP-001`;
    }

    if (g.referral) {
      badgeReferralStatus.textContent = "REFERABLE DR (YES)";
      badgeReferralStatus.className = "badge badge-referable";
    } else {
      badgeReferralStatus.textContent = "NON-REFERABLE (NO)";
      badgeReferralStatus.className = "badge badge-non-referable";
    }

    // 7. Probabilities Bar Chart (Pure EXP-001 Softmax Output)
    probBarsContainer.innerHTML = "";
    g.probabilities.forEach(p => {
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <span class="prob-label">Grade ${p.grade} (${p.label})</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: ${p.percentage}%; background: ${p.grade === g.grade ? getGradeColor(p.grade) : '#0969da'};"></div>
        </div>
        <span class="prob-pct">${p.percentage}%</span>
      `;
      probBarsContainer.appendChild(row);
    });
    probabilitiesSection.style.display = "block";

    // 8. Recommendation
    recTitle.textContent = g.referral_action;
    recBody.textContent = g.recommendation;

    // 9. Store Overlays
    currentOverlays = data.overlays || {};
    layerToggles.style.display = "flex";
    setActiveLayer("original");

    // 10. Report Links
    if (data.report) {
      setupReportLinks(data.report);
    }
  }

  function getGradeColor(grade) {
    const colors = ["#1a7f37", "#b08800", "#b08800", "#cf222e", "#82071e"];
    return colors[grade] || "#1f2328";
  }

  // Layer Toggles
  document.querySelectorAll(".toggle-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const layer = btn.getAttribute("data-layer");
      setActiveLayer(layer);
    });
  });

  function setActiveLayer(layer) {
    document.querySelectorAll(".toggle-btn").forEach(b => {
      b.classList.toggle("active", b.getAttribute("data-layer") === layer);
    });

    if (currentOverlays[layer]) {
      mainImageView.src = currentOverlays[layer];
    }

    // Update legend
    if (layer === "lesions") {
      layerLegend.style.display = "flex";
      layerLegend.innerHTML = `
        <span class="legend-item"><span class="legend-swatch" style="background:#ffe600;"></span> Hard Exudates (Disc Masked)</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#ff00ea;"></span> Microaneurysm Candidates</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#ff2a2a;"></span> Hemorrhages</span>
      `;
    } else if (layer === "heatmap") {
      layerLegend.style.display = "flex";
      layerLegend.innerHTML = `
        <span class="legend-item"><span class="legend-swatch" style="background:linear-gradient(to right, blue, cyan, yellow, red);"></span> Model Attention (Grad-CAM Saliency)</span>
        <small style="color:#656d76; margin-left:6px;">Highlights neural decision regions (Explainability, not lesion segmentation)</small>
      `;
    } else if (layer === "vessels") {
      layerLegend.style.display = "flex";
      layerLegend.innerHTML = `
        <span class="legend-item"><span class="legend-swatch" style="background:#2ea043;"></span> Segmented Vasculature</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#00dcff;"></span> Fovea</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#ffdc00;"></span> Optic Disc</span>
      `;
    } else {
      layerLegend.style.display = "none";
    }
  }

  function setupReportLinks(report) {
    reportActionBar.style.display = "flex";
    btnViewReport.href = report.relative_url;
    btnDownloadReport.href = `/api/download-report/${report.filename}`;
  }

  // About modal controls
  btnAbout.addEventListener("click", () => aboutModal.style.display = "flex");
  btnCloseAbout.addEventListener("click", () => aboutModal.style.display = "none");
  window.addEventListener("click", (e) => {
    if (e.target === aboutModal) aboutModal.style.display = "none";
  });
});
