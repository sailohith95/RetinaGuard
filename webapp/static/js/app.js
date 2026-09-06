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
  const probabilitiesSection = document.getElementById("probabilities-section");
  const probBarsContainer = document.getElementById("prob-bars-container");

  // Table
  const valOdStatus = document.getElementById("val-od-status");
  const valOdDetails = document.getElementById("val-od-details");
  const valFoveaStatus = document.getElementById("val-fovea-status");
  const valFoveaDetails = document.getElementById("val-fovea-details");
  const valVesselDensity = document.getElementById("val-vessel-density");
  const valMaCount = document.getElementById("val-ma-count");
  const valExCount = document.getElementById("val-ex-count");
  const valHeCount = document.getElementById("val-he-count");
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
        demoCases.forEach(c => {
          const opt = document.createElement("option");
          opt.value = c.id;
          opt.textContent = `${c.label} (${c.reference_name})`;
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

  // 2. Upload handling
  btnTriggerUpload.addEventListener("click", () => fileUpload.click());

  fileUpload.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    currentFile = file;
    currentDemoId = null;
    selectDemoCase.value = "";

    // Local Preview
    const objectUrl = URL.createObjectURL(file);
    showImage(objectUrl, file.name);

    // Update metadata badge
    const sizeKb = Math.round(file.size / 1024);
    imageMetaTag.textContent = `${file.name} (${sizeKb} KB)`;
    imageMetaTag.className = "badge badge-neutral";

    resetResults("Image loaded from file. Click Analyze Image to run screening.");
    btnAnalyze.disabled = false;
  });

  // 3. Demo Case loading
  btnLoadDemo.addEventListener("click", () => {
    const demoId = selectDemoCase.value;
    if (!demoId) return;

    const selectedCase = demoCases.find(c => c.id === demoId);
    if (!selectedCase) return;

    currentDemoId = demoId;
    currentFile = null;
    fileUpload.value = "";

    showImage(selectedCase.image_url, selectedCase.label);
    imageMetaTag.textContent = `${selectedCase.label}`;
    imageMetaTag.className = "badge badge-neutral";

    resetResults(`Loaded authentic reference case (${selectedCase.reference_name}). Click Analyze Image.`);
    btnAnalyze.disabled = false;
  });

  // Viewport image setter
  function showImage(src, alt) {
    emptyPlaceholder.style.display = "none";
    mainImageView.style.display = "block";
    mainImageView.src = src;
    mainImageView.alt = alt;
  }

  // Reset results
  function resetResults(msg) {
    layerToggles.style.display = "none";
    layerLegend.style.display = "none";
    safetyAlertBox.style.display = "none";
    borderlineAlertBox.style.display = "none";
    probabilitiesSection.style.display = "none";
    reportActionBar.style.display = "none";

    badgeQualityStatus.textContent = "PENDING";
    badgeQualityStatus.className = "badge badge-neutral";
    valQualityScore.textContent = "--";
    qualityMeterFill.style.width = "0%";
    valQFocus.textContent = "--";
    valQIllum.textContent = "--";
    valQContrast.textContent = "--";
    valQFov.textContent = "--";
    boxQualityReason.style.display = "none";

    badgeReferralStatus.textContent = "PENDING";
    badgeReferralStatus.className = "badge badge-neutral";
    textGradeHeadline.textContent = "Awaiting Analysis";
    textGradeConfidence.textContent = msg || "Ready to execute screening";

    valOdStatus.textContent = "--";
    valFoveaStatus.textContent = "--";
    valVesselDensity.textContent = "--";
    valMaCount.textContent = "--";
    valExCount.textContent = "--";
    valHeCount.textContent = "--";
    valNvRisk.textContent = "--";

    recTitle.textContent = "Standard Triage Recommendation";
    recBody.textContent = msg || "Upload or load an image to receive clinical guidance.";
  }

  // 4. Run Analysis
  btnAnalyze.addEventListener("click", async () => {
    if (!currentFile && !currentDemoId) return;

    btnAnalyze.disabled = true;
    analyzeSpinner.style.display = "inline-block";
    analyzeText.textContent = "Analyzing...";
    processingOverlay.style.display = "flex";

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

      currentAnalysis = await resp.json();
      renderAnalysisResults(currentAnalysis);
    } catch (err) {
      alert("Error: " + err.message);
      resetResults("Analysis error occurred.");
    } finally {
      btnAnalyze.disabled = false;
      analyzeSpinner.style.display = "none";
      analyzeText.textContent = "Analyze Image ▶";
      processingOverlay.style.display = "none";
    }
  });

  // Render Full Results
  function renderAnalysisResults(data) {
    const q = data.quality;

    // Quality Panel
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

    // CRITICAL SAFETY GATE HANDLING: UNGRADABLE
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
      textGradeConfidence.textContent = "Image quality is insufficient to produce a reliable diagnosis.";

      recTitle.textContent = "Safety Rejection Guidance";
      recBody.textContent = q.recommendation;

      // Enable report download for ungradable record
      if (data.report) {
        setupReportLinks(data.report);
      }
      return;
    }

    // Borderline Alert
    if (q.status === "BORDERLINE") {
      borderlineAlertBox.style.display = "block";
      borderlineAlertBox.innerHTML = `
        <strong>BORDERLINE QUALITY:</strong> Automated green-channel CLAHE enhancement was applied.<br>
        <em>${q.reason}</em>
      `;
    }

    // Landmarks & Lesions
    const st = data.structures;
    const l = data.lesions;

    valOdStatus.textContent = `Centroid [${st.optic_disc.centroid.join(", ")}]`;
    valOdDetails.textContent = `Radius: ${st.optic_disc.radius_px} px (Mask generated)`;
    valFoveaStatus.textContent = `Coords [${st.fovea.coordinates.join(", ")}]`;
    valVesselDensity.textContent = `${st.vessels.density_percent}%`;

    valMaCount.textContent = `${l.microaneurysms.candidate_count} candidates`;
    valExCount.textContent = `${l.hard_exudates.candidate_count} candidates (${l.hard_exudates.total_area_px} px)`;
    valHeCount.textContent = `${l.hemorrhages.candidate_count} candidates (${l.hemorrhages.total_area_px} px)`;
    valNvRisk.textContent = `${l.neovascularization.risk_level} Risk`;

    // DR Grading (EXP-001)
    const g = data.grading;
    textGradeHeadline.textContent = `Grade ${g.grade} — ${g.severity_name}`;
    textGradeHeadline.style.color = getGradeColor(g.grade);
    textGradeConfidence.textContent = `Confidence: ${g.confidence_percent}% (${g.confidence_level} Confidence) • Model: EXP-001`;

    if (g.referral) {
      badgeReferralStatus.textContent = "REFERABLE DR";
      badgeReferralStatus.className = "badge badge-referable";
    } else {
      badgeReferralStatus.textContent = "NON-REFERABLE";
      badgeReferralStatus.className = "badge badge-non-referable";
    }

    // Probabilities Bar Chart
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

    // Recommendation
    recTitle.textContent = g.referral_action;
    recBody.textContent = g.recommendation;

    // Store Overlays
    currentOverlays = data.overlays;
    layerToggles.style.display = "flex";
    setActiveLayer("original");

    // Report Links
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
