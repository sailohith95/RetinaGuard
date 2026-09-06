classdef RetinaGuardApp < matlab.apps.AppBase

% RetinaGuardApp  Main application class for the RetinaGuard DR Screening System.
%
%   Launch:
%       app = RetinaGuardApp;
%
%   Architecture:
%       Presentation  →  runScreeningPipeline()
%       Pipeline      →  quality / enhancement / structures / lesions / grading
%       Model layer   →  DRClassifierModel / heuristic placeholders
%
%   All dataset/model paths are in config/RGConfig.m.
%   Demo mode is active when no trained model is found.

    % ===================================================================
    % Properties — App-wide state
    % ===================================================================
    properties (Access = private)

        % ---- UI Windows & Root layout ---------------------------------
        UIFigure        matlab.ui.Figure

        % ---- Navigation ----------------------------------------------
        NavPanel        matlab.ui.container.Panel
        NavButtons      struct   % .Screening .Patients .Examinations .Reports .Model .System

        % ---- Main content container ----------------------------------
        ContentPanel    matlab.ui.container.Panel

        % ---- Panels for each module ----------------------------------
        PanelScreening      matlab.ui.container.Panel
        PanelPatients       matlab.ui.container.Panel
        PanelExaminations   matlab.ui.container.Panel
        PanelReports        matlab.ui.container.Panel
        PanelModel          matlab.ui.container.Panel
        PanelSystem         matlab.ui.container.Panel

        % ---- Screening UI components ---------------------------------
        % Exam header
        LblPatientID        matlab.ui.control.Label
        LblExamID           matlab.ui.control.Label
        LblDateTime         matlab.ui.control.Label
        LblEye              matlab.ui.control.Label
        LblStatus           matlab.ui.control.Label

        % Image workspace
        AxImage             matlab.ui.control.UIAxes
        BtnUpload           matlab.ui.control.Button
        BtnDemoLoad         matlab.ui.control.Button
        BtnAnalyze          matlab.ui.control.Button
        BtnToggleEnhanced   matlab.ui.control.Button
        BtnResetView        matlab.ui.control.Button
        DemoDropdown        matlab.ui.control.DropDown

        % Quality panel
        LblQualityStatus    matlab.ui.control.Label
        LblQualityScore     matlab.ui.control.Label
        LblFocus            matlab.ui.control.Label
        LblIllum            matlab.ui.control.Label
        LblFOV              matlab.ui.control.Label
        LblContrast         matlab.ui.control.Label

        % Processing progress
        PanelProgress       matlab.ui.container.Panel
        LblProgressStage    matlab.ui.control.Label
        ProgressBar         matlab.ui.control.Gauge

        % Lesion findings
        LblMA               matlab.ui.control.Label
        LblExudate          matlab.ui.control.Label
        LblHemorrhage       matlab.ui.control.Label
        LblNV               matlab.ui.control.Label

        % Structures
        LblOD               matlab.ui.control.Label
        LblFovea            matlab.ui.control.Label
        LblVessels          matlab.ui.control.Label

        % Overlay buttons
        BtnOverlayOrig      matlab.ui.control.Button
        BtnOverlayEnhanced  matlab.ui.control.Button
        BtnOverlayVessel    matlab.ui.control.Button
        BtnOverlayLesion    matlab.ui.control.Button
        BtnOverlayHeatmap   matlab.ui.control.Button

        % DR Result panel
        LblGradeNum         matlab.ui.control.Label
        LblGradeName        matlab.ui.control.Label
        LblConfidence       matlab.ui.control.Label
        LblReferral         matlab.ui.control.Label
        LblDemoTag          matlab.ui.control.Label

        % Severity scale
        AxSeverity          matlab.ui.control.UIAxes

        % Actions
        BtnReport           matlab.ui.control.Button
        BtnNewCase          matlab.ui.control.Button

        % ---- App state -----------------------------------------------
        cfg                         % RGConfig struct
        currentImage        = []    % uint8 RGB original
        currentEnhanced     = []    % uint8 RGB enhanced
        currentScreening    = []    % full pipeline result
        currentDemoCase     = []    % selected demo case struct
        demoCases           = {}    % all 6 demo cases
        showingEnhanced     = false
        currentOverlay      = 'original'
        activeNav           = 'Screening'
        patientCounter      = 0
        examCounter         = 0
        patientDB           = {}    % cell of patient structs
        reportPaths         = {}    % cell of generated report paths

        % ---- Patient/Report table UI --------------------------------
        PatientTable        matlab.ui.control.Table
        ExamTable           matlab.ui.control.Table
        ReportTable         matlab.ui.control.Table

        % ---- Model status UI ----------------------------------------
        LblDRModelStatus    matlab.ui.control.Label
        LblQualModelStatus  matlab.ui.control.Label
        LblLesionModelStatus matlab.ui.control.Label
        LblStructModelStatus matlab.ui.control.Label
        LblDatasetStatus    matlab.ui.control.Label
    end

    % ===================================================================
    % Colors / Design tokens
    % ===================================================================
    properties (Constant, Access = private)
        CLR_NAV_BG      = [0.106  0.165  0.271]   % #1b2a45
        CLR_NAV_ACTIVE  = [0.141  0.212  0.349]   % #243659
        CLR_HEADER_BG   = [0.106  0.165  0.271]
        CLR_PAGE_BG     = [0.957  0.965  0.973]   % #f4f6f8
        CLR_SURFACE     = [1      1      1    ]
        CLR_BORDER      = [0.878  0.898  0.918]   % #e0e6ea
        CLR_TEXT_PRI    = [0.102  0.141  0.200]   % #1a2433
        CLR_TEXT_SEC    = [0.420  0.478  0.565]   % #6b7a90
        CLR_ACCENT      = [0.106  0.361  0.659]   % #1b5ca8
        CLR_SUCCESS     = [0.176  0.478  0.176]   % #2d7a2d
        CLR_WARNING     = [0.769  0.490  0.000]   % #c47d00
        CLR_DANGER      = [0.710  0.188  0.000]   % #b53000
        CLR_PDR         = [0.545  0.000  0.000]   % #8b0000
        CLR_DEMO_BG     = [0.478  0.310  0.000]   % #7a4f00
        CLR_DEMO_FG     = [1      0.969  0.902]   % #fff7e6
    end

    % ===================================================================
    % App creation
    % ===================================================================
    methods (Access = public)

        function app = RetinaGuardApp()
            % Add RetinaGuard root to path
            rootDir = fileparts(fileparts(mfilename('fullpath')));
            addpath(genpath(rootDir));

            % Load config
            app.cfg = RGConfig();

            % Load demo cases
            try
                app.demoCases = getDemoCases();
            catch
                app.demoCases = {};
            end

            % Build UI
            app.buildUI();
            app.navigateTo('Screening');

            % Register with App framework
            registerApp(app, app.UIFigure);

            if nargout == 0, clear app; end
        end

        function delete(app)
            delete(app.UIFigure);
        end
    end

    % ===================================================================
    % UI Construction
    % ===================================================================
    methods (Access = private)

        function buildUI(app)
            % ---- Main window ----------------------------------------
            app.UIFigure = uifigure( ...
                'Name',     'RetinaGuard — AI-Assisted DR Screening', ...
                'Position', [80 60 1380 820], ...
                'Color',    app.CLR_PAGE_BG, ...
                'Resize',   'on');
            app.UIFigure.CloseRequestFcn = @(~,~) delete(app);

            % ---- Header bar -----------------------------------------
            header = uipanel(app.UIFigure, ...
                'Position',         [0 790 1380 30], ...
                'BackgroundColor',  app.CLR_HEADER_BG, ...
                'BorderType',       'none');
            uilabel(header, 'Text', 'RetinaGuard', ...
                'Position', [14 5 160 20], ...
                'FontSize', 14, 'FontWeight', 'bold', ...
                'FontColor', [1 1 1]);
            uilabel(header, 'Text', 'AI-Assisted DR Screening', ...
                'Position', [130 6 220 18], ...
                'FontSize', 11, 'FontColor', [0.65 0.75 0.88]);
            uilabel(header, 'Text', ['v' app.cfg.app.version '  |  SIH 2026 Prototype'], ...
                'Position', [1180 6 185 18], ...
                'FontSize', 10, 'FontColor', [0.55 0.65 0.78], ...
                'HorizontalAlignment', 'right');

            % ---- Navigation sidebar ---------------------------------
            app.NavPanel = uipanel(app.UIFigure, ...
                'Position',        [0 0 160 790], ...
                'BackgroundColor', app.CLR_NAV_BG, ...
                'BorderType',      'none');

            % Logo area
            logoPanel = uipanel(app.NavPanel, ...
                'Position', [0 720 160 70], ...
                'BackgroundColor', [0.082 0.133 0.220], ...
                'BorderType', 'none');
            uilabel(logoPanel, 'Text', char(hex2dec('25CB')), ...
                'Position', [14 30 26 30], ...
                'FontSize', 22, 'FontColor', [0.2 0.7 1.0]);
            uilabel(logoPanel, 'Text', 'RetinaGuard', ...
                'Position', [42 36 110 22], ...
                'FontSize', 12, 'FontWeight', 'bold', 'FontColor', [1 1 1]);
            uilabel(logoPanel, 'Text', 'DR Screening System', ...
                'Position', [42 18 110 18], ...
                'FontSize', 9, 'FontColor', [0.55 0.65 0.78]);

            navItems = {'Screening','Patients','Examinations','Reports','Model / AI','System'};
            navKeys  = {'Screening','Patients','Examinations','Reports','Model','System'};
            navIcons = {char(hex2dec('25A3')), char(hex2dec('25A1')), char(hex2dec('25A1')), char(hex2dec('25A1')), char(hex2dec('25A1')), char(hex2dec('25A1'))};

            for i = 1:numel(navItems)
                ypos = 720 - i*52;
                btn = uibutton(app.NavPanel, 'push', ...
                    'Text',            ['  ' navItems{i}], ...
                    'Position',        [0 ypos 160 44], ...
                    'FontSize',        12, ...
                    'FontColor',       [0.7 0.8 0.92], ...
                    'BackgroundColor', app.CLR_NAV_BG, ...
                    'HorizontalAlignment', 'left');
                key = navKeys{i};
                btn.ButtonPushedFcn = @(~,~) app.navigateTo(key);
                app.NavButtons.(key) = btn;
            end

            % ---- Content area ---------------------------------------
            app.ContentPanel = uipanel(app.UIFigure, ...
                'Position',        [160 0 1220 790], ...
                'BackgroundColor', app.CLR_PAGE_BG, ...
                'BorderType',      'none');

            % Build all module panels
            app.buildScreeningPanel();
            app.buildPatientsPanel();
            app.buildExaminationsPanel();
            app.buildReportsPanel();
            app.buildModelPanel();
            app.buildSystemPanel();
        end

        % ===============================================================
        %  SCREENING PANEL
        % ===============================================================
        function buildScreeningPanel(app)
            p = uipanel(app.ContentPanel, ...
                'Position', [0 0 1220 790], ...
                'BackgroundColor', app.CLR_PAGE_BG, ...
                'BorderType', 'none', 'Visible', 'off');
            app.PanelScreening = p;

            % ---- Section title bar ----------------------------------
            titleBar = uipanel(p, 'Position', [0 748 1220 42], ...
                'BackgroundColor', app.CLR_SURFACE, 'BorderType', 'line');
            uilabel(titleBar, 'Text', 'Retinal Screening Workstation', ...
                'Position', [16 8 400 24], ...
                'FontSize', 15, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_PRI);

            % ---- Exam header bar ------------------------------------
            examBar = uipanel(p, 'Position', [0 698 1220 50], ...
                'BackgroundColor', app.CLR_SURFACE, 'BorderType', 'line');

            uilabel(examBar, 'Text', 'PATIENT', 'Position', [14 32 70 14], 'FontSize', 9, 'FontColor', app.CLR_TEXT_SEC);
            app.LblPatientID = uilabel(examBar, 'Text', '—', 'Position', [14 12 100 18], 'FontSize', 12, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_PRI);

            uilabel(examBar, 'Text', 'EXAM ID', 'Position', [128 32 80 14], 'FontSize', 9, 'FontColor', app.CLR_TEXT_SEC);
            app.LblExamID = uilabel(examBar, 'Text', '—', 'Position', [128 12 120 18], 'FontSize', 12, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_PRI);

            uilabel(examBar, 'Text', 'DATE / TIME', 'Position', [268 32 100 14], 'FontSize', 9, 'FontColor', app.CLR_TEXT_SEC);
            app.LblDateTime = uilabel(examBar, 'Text', datestr(now,'dd-mmm-yyyy HH:MM'), 'Position', [268 12 160 18], 'FontSize', 12, 'FontColor', app.CLR_TEXT_PRI);

            uilabel(examBar, 'Text', 'EYE', 'Position', [442 32 60 14], 'FontSize', 9, 'FontColor', app.CLR_TEXT_SEC);
            app.LblEye = uilabel(examBar, 'Text', 'Unknown', 'Position', [442 12 80 18], 'FontSize', 12, 'FontColor', app.CLR_TEXT_PRI);

            uilabel(examBar, 'Text', 'STATUS', 'Position', [536 32 80 14], 'FontSize', 9, 'FontColor', app.CLR_TEXT_SEC);
            app.LblStatus = uilabel(examBar, 'Text', 'Not Analyzed', 'Position', [536 12 150 18], 'FontSize', 12, 'FontColor', app.CLR_TEXT_SEC);

            % ---- Demo mode banner -----------------------------------
            app.LblDemoTag = uilabel(p, 'Text', '⚠  DEMO MODE — Demonstration pipeline active. Results are not medically validated.', ...
                'Position', [0 672 1220 26], ...
                'FontSize', 11, 'FontWeight', 'bold', ...
                'FontColor', app.CLR_DEMO_FG, ...
                'BackgroundColor', app.CLR_DEMO_BG, ...
                'HorizontalAlignment', 'center');

            % ==========================================================
            %  LEFT COLUMN — Image workspace  (x=8, width≈680)
            % ==========================================================
            imgPanel = uipanel(p, 'Title', '', ...
                'Position', [8 8 680 660], ...
                'BackgroundColor', app.CLR_SURFACE, 'BorderType', 'line');

            % Control bar above image
            ctrlBar = uipanel(imgPanel, 'Position', [0 612 680 40], ...
                'BackgroundColor', [0.95 0.96 0.97], 'BorderType', 'none');

            app.BtnUpload = uibutton(ctrlBar, 'push', 'Text', 'Upload Image', ...
                'Position', [8 6 110 28], 'FontSize', 11, ...
                'BackgroundColor', app.CLR_ACCENT, 'FontColor', [1 1 1]);
            app.BtnUpload.ButtonPushedFcn = @(~,~) app.onUploadImage();

            uilabel(ctrlBar, 'Text', 'Demo:', 'Position', [128 10 38 20], 'FontSize', 10, 'FontColor', app.CLR_TEXT_SEC);
            demoLabels = {'Select demo case...'};
            for k = 1:numel(app.demoCases)
                demoLabels{end+1} = app.demoCases{k}.label;
            end
            app.DemoDropdown = uidropdown(ctrlBar, 'Items', demoLabels, ...
                'Position', [168 8 220 24], 'FontSize', 10);
            app.BtnDemoLoad = uibutton(ctrlBar, 'push', 'Text', 'Load Demo', ...
                'Position', [396 8 80 24], 'FontSize', 10, ...
                'BackgroundColor', [0.92 0.94 0.96]);
            app.BtnDemoLoad.ButtonPushedFcn = @(~,~) app.onLoadDemo();

            app.BtnAnalyze = uibutton(ctrlBar, 'push', 'Text', 'Analyze Image ▶', ...
                'Position', [490 6 120 28], 'FontSize', 11, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.141 0.439 0.141], 'FontColor', [1 1 1]);
            app.BtnAnalyze.ButtonPushedFcn = @(~,~) app.onAnalyze();

            % Overlay toggle bar
            overlayBar = uipanel(imgPanel, 'Position', [0 578 680 34], ...
                'BackgroundColor', [0.93 0.945 0.96], 'BorderType', 'none');
            uilabel(overlayBar, 'Text', 'View:', 'Position', [8 7 36 20], 'FontSize', 10, 'FontColor', app.CLR_TEXT_SEC);

            overlayBtns = {'Original','Enhanced','Vessels','Lesions','Heatmap'};
            overlayKeys = {'original','enhanced','vessel','lesion','heatmap'};
            xpos = 50;
            for i = 1:numel(overlayBtns)
                b = uibutton(overlayBar, 'push', 'Text', overlayBtns{i}, ...
                    'Position', [xpos 5 72 24], 'FontSize', 9, ...
                    'BackgroundColor', [0.88 0.90 0.93]);
                key = overlayKeys{i};
                b.ButtonPushedFcn = @(~,~) app.onSetOverlay(key);
                xpos = xpos + 78;
            end

            % Main image axes
            app.AxImage = uiaxes(imgPanel, ...
                'Position', [6 6 668 572], ...
                'BackgroundColor', [0.12 0.12 0.15]);
            app.AxImage.XAxis.Visible = 'off';
            app.AxImage.YAxis.Visible = 'off';
            app.AxImage.Toolbar.Visible = 'off';
            app.AxImage.Box = 'off';
            disableDefaultInteractivity(app.AxImage);

            % Placeholder text when no image
            text(app.AxImage, 0.5, 0.5, ...
                {'Upload a retinal fundus image', 'or load a demo case to begin'}, ...
                'Units','normalized','HorizontalAlignment','center', ...
                'FontSize', 13, 'Color', [0.5 0.55 0.6]);

            % ==========================================================
            %  RIGHT COLUMN (x=696, width≈516)
            % ==========================================================

            % ---- Image Quality panel (top right) ----------------------
            qPanel = uipanel(p, 'Title', 'IMAGE QUALITY', ...
                'Position', [696 530 516 138], ...
                'BackgroundColor', app.CLR_SURFACE, 'BorderType', 'line', ...
                'FontSize', 9, 'FontWeight', 'bold', 'ForegroundColor', app.CLR_TEXT_SEC);

            uilabel(qPanel, 'Text', 'Overall', 'Position', [10 92 60 14], 'FontSize', 9, 'FontColor', app.CLR_TEXT_SEC);
            app.LblQualityStatus = uilabel(qPanel, 'Text', '—', 'Position', [10 70 120 22], 'FontSize', 13, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_SEC);

            uilabel(qPanel, 'Text', 'Score', 'Position', [140 92 60 14], 'FontSize', 9, 'FontColor', app.CLR_TEXT_SEC);
            app.LblQualityScore = uilabel(qPanel, 'Text', '— / 100', 'Position', [140 70 90 22], 'FontSize', 13, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_PRI);

            % Quality metrics grid
            metricLabels = {'Focus','Illumination','Field of View','Contrast'};
            metricProps  = {'LblFocus','LblIllum','LblFOV','LblContrast'};
            cols = [280 380]; rows2 = [88 58];
            idx = 1;
            for r = rows2
                for c = cols
                    if idx > numel(metricLabels), break; end
                    uilabel(qPanel, 'Text', metricLabels{idx}, ...
                        'Position', [c-280+280 r+14 100 12], 'FontSize', 9, 'FontColor', app.CLR_TEXT_SEC);
                    lbl = uilabel(qPanel, 'Text', '—', ...
                        'Position', [c-280+280 r 100 16], 'FontSize', 11, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_PRI);
                    app.(metricProps{idx}) = lbl;
                    idx = idx + 1;
                end
            end
            % Re-layout quality metrics simply
            app.LblFocus   = uilabel(qPanel, 'Text', '—', 'Position', [270 90 90 16], 'FontSize', 11, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_PRI);
            uilabel(qPanel, 'Text', 'Focus',  'Position', [270 104 90 12], 'FontSize', 9, 'FontColor', app.CLR_TEXT_SEC);
            app.LblIllum   = uilabel(qPanel, 'Text', '—', 'Position', [370 90 110 16], 'FontSize', 11, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_PRI);
            uilabel(qPanel, 'Text', 'Illumination', 'Position', [370 104 110 12], 'FontSize', 9, 'FontColor', app.CLR_TEXT_SEC);
            app.LblFOV     = uilabel(qPanel, 'Text', '—', 'Position', [270 60 90 16], 'FontSize', 11, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_PRI);
            uilabel(qPanel, 'Text', 'Field of View', 'Position', [270 72 110 12], 'FontSize', 9, 'FontColor', app.CLR_TEXT_SEC);
            app.LblContrast = uilabel(qPanel, 'Text', '—', 'Position', [370 60 110 16], 'FontSize', 11, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_PRI);
            uilabel(qPanel, 'Text', 'Contrast', 'Position', [370 72 110 12], 'FontSize', 9, 'FontColor', app.CLR_TEXT_SEC);

            % ---- Lesion Analysis panel -------------------------------
            lPanel = uipanel(p, 'Title', 'LESION ANALYSIS', ...
                'Position', [696 380 250 148], ...
                'BackgroundColor', app.CLR_SURFACE, 'BorderType', 'line', ...
                'FontSize', 9, 'FontWeight', 'bold', 'ForegroundColor', app.CLR_TEXT_SEC);

            lesionLabels = {'Microaneurysms','Exudates','Hemorrhages','Neovascularization'};
            lesionProps  = {'LblMA','LblExudate','LblHemorrhage','LblNV'};
            for i = 1:4
                yp = 108 - (i-1)*26;
                uilabel(lPanel, 'Text', lesionLabels{i}, 'Position', [8 yp+10 150 12], 'FontSize', 9, 'FontColor', app.CLR_TEXT_SEC);
                lbl = uilabel(lPanel, 'Text', '—', 'Position', [8 yp 200 14], 'FontSize', 10, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_PRI);
                app.(lesionProps{i}) = lbl;
            end

            % ---- Retinal Structures panel ---------------------------
            sPanel = uipanel(p, 'Title', 'RETINAL STRUCTURES', ...
                'Position', [954 380 258 148], ...
                'BackgroundColor', app.CLR_SURFACE, 'BorderType', 'line', ...
                'FontSize', 9, 'FontWeight', 'bold', 'ForegroundColor', app.CLR_TEXT_SEC);

            structLabels = {'Optic Disc','Fovea','Blood Vessels'};
            structProps  = {'LblOD','LblFovea','LblVessels'};
            for i = 1:3
                yp = 108 - (i-1)*34;
                uilabel(sPanel, 'Text', structLabels{i}, 'Position', [8 yp+12 130 12], 'FontSize', 9, 'FontColor', app.CLR_TEXT_SEC);
                lbl = uilabel(sPanel, 'Text', '—', 'Position', [8 yp 220 16], 'FontSize', 11, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_PRI);
                app.(structProps{i}) = lbl;
            end

            % ---- DR Result panel ------------------------------------
            rPanel = uipanel(p, 'Title', 'AI-ASSISTED SCREENING RESULT', ...
                'Position', [696 130 516 248], ...
                'BackgroundColor', app.CLR_SURFACE, 'BorderType', 'line', ...
                'FontSize', 9, 'FontWeight', 'bold', 'ForegroundColor', app.CLR_TEXT_SEC);

            uilabel(rPanel, 'Text', 'DR SEVERITY', 'Position', [14 198 150 14], 'FontSize', 9, 'FontColor', app.CLR_TEXT_SEC, 'FontWeight', 'bold');
            app.LblGradeNum = uilabel(rPanel, 'Text', '—', 'Position', [14 148 120 50], 'FontSize', 44, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_SEC);
            app.LblGradeName = uilabel(rPanel, 'Text', 'Not analyzed', 'Position', [14 128 400 22], 'FontSize', 13, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_PRI);
            app.LblConfidence = uilabel(rPanel, 'Text', 'Model confidence: —', 'Position', [14 108 300 18], 'FontSize', 11, 'FontColor', app.CLR_TEXT_SEC);

            % Severity scale axes
            app.AxSeverity = uiaxes(rPanel, 'Position', [8 60 496 44], ...
                'BackgroundColor', [0.95 0.96 0.97]);
            app.AxSeverity.XAxis.Visible = 'off';
            app.AxSeverity.YAxis.Visible = 'off';
            app.AxSeverity.Toolbar.Visible = 'off';
            disableDefaultInteractivity(app.AxSeverity);
            app.drawSeverityScale(-1);

            % Referral recommendation
            app.LblReferral = uilabel(rPanel, 'Text', 'Upload and analyze a retinal image to begin screening.', ...
                'Position', [8 6 498 52], 'FontSize', 11, 'FontColor', app.CLR_TEXT_PRI, ...
                'WordWrap', 'on');

            % ---- Action buttons panel -------------------------------
            aPanel = uipanel(p, 'Position', [696 8 516 120], ...
                'BackgroundColor', app.CLR_SURFACE, 'BorderType', 'line');

            uilabel(aPanel, 'Text', 'ACTIONS', 'Position', [14 96 100 14], 'FontSize', 9, 'FontColor', app.CLR_TEXT_SEC, 'FontWeight', 'bold');

            app.BtnReport = uibutton(aPanel, 'push', 'Text', 'Generate Report', ...
                'Position', [14 60 160 34], 'FontSize', 12, ...
                'BackgroundColor', app.CLR_ACCENT, 'FontColor', [1 1 1]);
            app.BtnReport.ButtonPushedFcn = @(~,~) app.onGenerateReport();

            app.BtnNewCase = uibutton(aPanel, 'push', 'Text', 'New Examination', ...
                'Position', [184 60 160 34], 'FontSize', 12, ...
                'BackgroundColor', [0.92 0.94 0.96], 'FontColor', app.CLR_TEXT_PRI);
            app.BtnNewCase.ButtonPushedFcn = @(~,~) app.onNewCase();

            uilabel(aPanel, 'Text', ...
                'AI-assisted screening result only. Not a substitute for ophthalmologist evaluation.', ...
                'Position', [10 10 494 38], 'FontSize', 9.5, ...
                'FontColor', [0.65 0.55 0.35], 'WordWrap', 'on');

            % ---- Processing progress overlay (hidden initially) -----
            app.PanelProgress = uipanel(p, 'Position', [200 250 820 300], ...
                'BackgroundColor', app.CLR_SURFACE, 'BorderType', 'line', ...
                'Visible', 'off');
            uilabel(app.PanelProgress, 'Text', 'PROCESSING', ...
                'Position', [0 248 820 28], 'FontSize', 10, 'FontWeight', 'bold', ...
                'FontColor', app.CLR_TEXT_SEC, 'HorizontalAlignment', 'center');
            app.LblProgressStage = uilabel(app.PanelProgress, ...
                'Text', 'Initializing...', ...
                'Position', [0 216 820 28], 'FontSize', 14, ...
                'FontColor', app.CLR_TEXT_PRI, 'HorizontalAlignment', 'center');
            app.ProgressBar = uigauge(app.PanelProgress, 'linear', ...
                'Position', [80 165 660 36], 'Limits', [0 100], 'Value', 0, ...
                'MajorTicks', [], 'MinorTicks', [], ...
                'BackgroundColor', [0.92 0.94 0.96]);

            stagesList = {'Quality Check','Enhancement','Structures','Lesions','Classification','Report'};
            for i = 1:numel(stagesList)
                xp = 65 + (i-1)*118;
                uilabel(app.PanelProgress, 'Text', stagesList{i}, ...
                    'Position', [xp 130 100 20], 'FontSize', 9, 'FontColor', app.CLR_TEXT_SEC, ...
                    'HorizontalAlignment', 'center');
            end
        end

        % ===============================================================
        %  PATIENTS PANEL
        % ===============================================================
        function buildPatientsPanel(app)
            p = uipanel(app.ContentPanel, ...
                'Position', [0 0 1220 790], ...
                'BackgroundColor', app.CLR_PAGE_BG, ...
                'BorderType', 'none', 'Visible', 'off');
            app.PanelPatients = p;

            titleBar = uipanel(p, 'Position', [0 748 1220 42], ...
                'BackgroundColor', app.CLR_SURFACE, 'BorderType', 'line');
            uilabel(titleBar, 'Text', 'Patient Registry', ...
                'Position', [16 8 300 24], 'FontSize', 15, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_PRI);

            tablePanel = uipanel(p, 'Position', [8 8 1204 730], ...
                'BackgroundColor', app.CLR_SURFACE, 'BorderType', 'line');
            uilabel(tablePanel, 'Text', 'No patients registered in this session. Begin a screening examination to add patients.', ...
                'Position', [20 340 1160 40], 'FontSize', 12, 'FontColor', app.CLR_TEXT_SEC, 'HorizontalAlignment', 'center');
            uilabel(tablePanel, 'Text', 'Patient data is session-only in the prototype. Full EMR integration can be added later.', ...
                'Position', [20 310 1160 30], 'FontSize', 10, 'FontColor', app.CLR_TEXT_SEC, 'HorizontalAlignment', 'center');

            app.PatientTable = uitable(tablePanel, ...
                'Position', [10 10 1180 700], ...
                'ColumnName', {'Patient ID','Name','Age','Sex','Last Exam','Latest Grade','Status'}, ...
                'ColumnWidth', {120 200 60 60 160 160 120}, ...
                'RowName', {}, 'FontSize', 11, 'Visible', 'off');
        end

        % ===============================================================
        %  EXAMINATIONS PANEL
        % ===============================================================
        function buildExaminationsPanel(app)
            p = uipanel(app.ContentPanel, ...
                'Position', [0 0 1220 790], ...
                'BackgroundColor', app.CLR_PAGE_BG, ...
                'BorderType', 'none', 'Visible', 'off');
            app.PanelExaminations = p;

            titleBar = uipanel(p, 'Position', [0 748 1220 42], ...
                'BackgroundColor', app.CLR_SURFACE, 'BorderType', 'line');
            uilabel(titleBar, 'Text', 'Examination History', ...
                'Position', [16 8 300 24], 'FontSize', 15, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_PRI);

            tablePanel = uipanel(p, 'Position', [8 8 1204 730], ...
                'BackgroundColor', app.CLR_SURFACE, 'BorderType', 'line');
            uilabel(tablePanel, ...
                'Text', 'No examinations recorded in this session. Examinations appear here after analysis.', ...
                'Position', [20 350 1160 30], 'FontSize', 12, 'FontColor', app.CLR_TEXT_SEC, 'HorizontalAlignment', 'center');
            app.ExamTable = uitable(tablePanel, ...
                'Position', [10 10 1180 700], ...
                'ColumnName', {'Exam ID','Patient ID','Date','Eye','Quality','DR Grade','Status'}, ...
                'ColumnWidth', {120 120 160 70 120 160 120}, ...
                'RowName', {}, 'FontSize', 11, 'Visible', 'off');
        end

        % ===============================================================
        %  REPORTS PANEL
        % ===============================================================
        function buildReportsPanel(app)
            p = uipanel(app.ContentPanel, ...
                'Position', [0 0 1220 790], ...
                'BackgroundColor', app.CLR_PAGE_BG, ...
                'BorderType', 'none', 'Visible', 'off');
            app.PanelReports = p;

            titleBar = uipanel(p, 'Position', [0 748 1220 42], ...
                'BackgroundColor', app.CLR_SURFACE, 'BorderType', 'line');
            uilabel(titleBar, 'Text', 'Screening Reports', ...
                'Position', [16 8 300 24], 'FontSize', 15, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_PRI);

            tablePanel = uipanel(p, 'Position', [8 8 1204 730], ...
                'BackgroundColor', app.CLR_SURFACE, 'BorderType', 'line');
            uilabel(tablePanel, ...
                'Text', 'No reports generated in this session. Generate a report from the Screening module.', ...
                'Position', [20 350 1160 30], 'FontSize', 12, 'FontColor', app.CLR_TEXT_SEC, 'HorizontalAlignment', 'center');
            app.ReportTable = uitable(tablePanel, ...
                'Position', [10 10 1180 700], ...
                'ColumnName', {'Report ID','Patient ID','Exam ID','Date','Grade','Quality','Path'}, ...
                'ColumnWidth', {120 120 120 160 160 120 320}, ...
                'RowName', {}, 'FontSize', 11, 'Visible', 'off');
        end

        % ===============================================================
        %  MODEL / AI PANEL
        % ===============================================================
        function buildModelPanel(app)
            p = uipanel(app.ContentPanel, ...
                'Position', [0 0 1220 790], ...
                'BackgroundColor', app.CLR_PAGE_BG, ...
                'BorderType', 'none', 'Visible', 'off');
            app.PanelModel = p;

            titleBar = uipanel(p, 'Position', [0 748 1220 42], ...
                'BackgroundColor', app.CLR_SURFACE, 'BorderType', 'line');
            uilabel(titleBar, 'Text', 'Model Status & AI Pipeline', ...
                'Position', [16 8 400 24], 'FontSize', 15, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_PRI);

            % Model status cards
            modelNames    = {'DR Severity Classifier','Image Quality Model','Lesion Detection Model','Structure Detection Model'};
            modelPaths    = {'models/dr/dr_classifier.mat','models/quality/quality_model.mat','models/lesions/lesion_model.mat','models/structures/structure_model.mat'};
            modelDescriptions = { ...
                'APTOS 2019 fine-tuned CNN — Classifies retinal image into DR grade 0-4', ...
                'Assesses image quality: focus, illumination, field of view', ...
                'IDRiD-trained segmentation — microaneurysms, exudates, hemorrhages, NV', ...
                'Optic disc localization, fovea detection, vessel segmentation'};
            modelProps = {'LblDRModelStatus','LblQualModelStatus','LblLesionModelStatus','LblStructModelStatus'};

            cfg_ = app.cfg;
            allPaths = {cfg_.paths.drModel, cfg_.paths.qualityModel, cfg_.paths.lesionModel, cfg_.paths.structModel};

            for i = 1:4
                yp = 650 - (i-1)*148;
                card = uipanel(p, 'Position', [8 yp 800 138], ...
                    'BackgroundColor', app.CLR_SURFACE, 'BorderType', 'line');
                uilabel(card, 'Text', modelNames{i}, ...
                    'Position', [14 102 600 22], 'FontSize', 13, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_PRI);
                uilabel(card, 'Text', modelDescriptions{i}, ...
                    'Position', [14 78 760 22], 'FontSize', 10, 'FontColor', app.CLR_TEXT_SEC);
                uilabel(card, 'Text', 'Status:', 'Position', [14 54 50 18], 'FontSize', 10, 'FontColor', app.CLR_TEXT_SEC);
                if isfile(allPaths{i})
                    statusTxt  = 'Trained Model Loaded';
                    statusClr  = app.CLR_SUCCESS;
                else
                    statusTxt  = 'Not Loaded — Demo Mode Active';
                    statusClr  = app.CLR_WARNING;
                end
                lbl = uilabel(card, 'Text', statusTxt, ...
                    'Position', [68 54 400 18], 'FontSize', 10, 'FontWeight', 'bold', 'FontColor', statusClr);
                app.(modelProps{i}) = lbl;
                uilabel(card, 'Text', ['Path: ' modelPaths{i}], ...
                    'Position', [14 28 760 16], 'FontSize', 9, 'FontColor', [0.6 0.65 0.7]);
                uilabel(card, 'Text', 'Performance metrics will display here when a trained model is loaded and evaluated.', ...
                    'Position', [14 8 760 16], 'FontSize', 9, 'FontColor', [0.7 0.6 0.4]);
            end

            % Dataset status
            dsPanel = uipanel(p, 'Title', 'DATASET STATUS', ...
                'Position', [820 500 390 288], ...
                'BackgroundColor', app.CLR_SURFACE, 'BorderType', 'line', ...
                'FontSize', 9, 'FontWeight', 'bold', 'ForegroundColor', app.CLR_TEXT_SEC);

            datasets = {'APTOS 2019 (DR Classification)', 'IDRiD (Lesion Segmentation)', 'Validation Dataset'};
            dataPaths = {cfg_.paths.aptos, cfg_.paths.idrid, fullfile(cfg_.paths.data,'validation')};
            for i = 1:3
                yp = 220 - (i-1)*70;
                uilabel(dsPanel, 'Text', datasets{i}, 'Position', [10 yp 300 18], 'FontSize', 11, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_PRI);
                if isfolder(dataPaths{i}) && ~isempty(dir(fullfile(dataPaths{i},'*')))
                    dsTxt = 'Connected'; dsClr = app.CLR_SUCCESS;
                else
                    dsTxt = 'Not Connected — Add data to data/ folder'; dsClr = app.CLR_TEXT_SEC;
                end
                uilabel(dsPanel, 'Text', dsTxt, 'Position', [10 yp-20 360 16], 'FontSize', 10, 'FontColor', dsClr);
                uilabel(dsPanel, 'Text', ['Path: ' strrep(dataPaths{i}, cfg_.paths.root, '.')], ...
                    'Position', [10 yp-36 360 14], 'FontSize', 9, 'FontColor', [0.65 0.70 0.75]);
            end

            uilabel(dsPanel, 'Text', 'See DATASET_SETUP.md for connection instructions.', ...
                'Position', [10 10 360 14], 'FontSize', 9, 'FontColor', [0.6 0.5 0.35]);
        end

        % ===============================================================
        %  SYSTEM PANEL
        % ===============================================================
        function buildSystemPanel(app)
            p = uipanel(app.ContentPanel, ...
                'Position', [0 0 1220 790], ...
                'BackgroundColor', app.CLR_PAGE_BG, ...
                'BorderType', 'none', 'Visible', 'off');
            app.PanelSystem = p;

            titleBar = uipanel(p, 'Position', [0 748 1220 42], ...
                'BackgroundColor', app.CLR_SURFACE, 'BorderType', 'line');
            uilabel(titleBar, 'Text', 'System Information', ...
                'Position', [16 8 300 24], 'FontSize', 15, 'FontWeight', 'bold', 'FontColor', app.CLR_TEXT_PRI);

            infoPanel = uipanel(p, 'Position', [8 400 600 338], ...
                'BackgroundColor', app.CLR_SURFACE, 'BorderType', 'line');

            infoItems = { ...
                'Application',     'RetinaGuard v1.0.0 Prototype'; ...
                'Purpose',         'AI-Assisted Diabetic Retinopathy Screening'; ...
                'Competition',     'Smart India Hackathon (SIH) 2026'; ...
                'Platform',        'MATLAB (desktop-first, offline)'; ...
                'Mode',            'Demo Mode (no real dataset connected)'; ...
                'Image Formats',   'JPG, JPEG, PNG, TIF, TIFF'; ...
                'Pipeline Stages', '5 (Quality → Enhancement → Structures → Lesions → Grade)'};

            for i = 1:size(infoItems,1)
                yp = 290 - (i-1)*38;
                uilabel(infoPanel, 'Text', infoItems{i,1}, 'Position', [14 yp+16 180 14], 'FontSize', 9, 'FontColor', app.CLR_TEXT_SEC);
                uilabel(infoPanel, 'Text', infoItems{i,2}, 'Position', [14 yp 540 18], 'FontSize', 11, 'FontColor', app.CLR_TEXT_PRI);
            end

            disclaimerPanel = uipanel(p, 'Position', [8 8 1204 385], ...
                'BackgroundColor', [0.99 0.97 0.93], 'BorderType', 'line');
            uilabel(disclaimerPanel, 'Text', 'IMPORTANT CLINICAL DISCLAIMER', ...
                'Position', [14 340 600 22], 'FontSize', 12, 'FontWeight', 'bold', 'FontColor', [0.6 0.35 0.0]);
            disclaimerText = [ ...
                'RetinaGuard is a prototype screening tool developed for research and demonstration purposes.' char(10) ...
                'It has NOT been evaluated in clinical trials, received regulatory approval, or undergone medical device certification.' char(10) char(10) ...
                'All screening results generated by this system are AI-assisted approximations based on image processing heuristics' char(10) ...
                'and/or placeholder models. They are NOT a substitute for examination by a qualified ophthalmologist.' char(10) char(10) ...
                'Demo results are clearly marked DEMO RESULT throughout the interface.' char(10) ...
                'No real patient data should be entered into this prototype system.' char(10) char(10) ...
                'Performance metrics (sensitivity, specificity, AUC) will only be displayed after proper clinical validation with real datasets.'];
            uilabel(disclaimerPanel, 'Text', disclaimerText, ...
                'Position', [14 10 1170 320], 'FontSize', 11, 'FontColor', [0.3 0.2 0.0], 'WordWrap', 'on');
        end
    end

    % ===================================================================
    % Navigation
    % ===================================================================
    methods (Access = private)

        function navigateTo(app, section)
            app.activeNav = section;
            panels = {'Screening','Patients','Examinations','Reports','Model','System'};
            panelProps = {'PanelScreening','PanelPatients','PanelExaminations','PanelReports','PanelModel','PanelSystem'};

            for i = 1:numel(panels)
                app.(panelProps{i}).Visible = 'off';
                if isfield(app.NavButtons, panels{i})
                    btn = app.NavButtons.(panels{i});
                    btn.BackgroundColor = app.CLR_NAV_BG;
                    btn.FontColor = [0.7 0.8 0.92];
                end
            end

            % Show selected panel
            idx = find(strcmp(panels, section));
            if ~isempty(idx)
                app.(panelProps{idx}).Visible = 'on';
            end

            % Highlight nav button
            if isfield(app.NavButtons, section)
                app.NavButtons.(section).BackgroundColor = app.CLR_NAV_ACTIVE;
                app.NavButtons.(section).FontColor = [1 1 1];
            end
        end
    end

    % ===================================================================
    % Screening Callbacks
    % ===================================================================
    methods (Access = private)

        % ---- Upload image --------------------------------------------
        function onUploadImage(app)
            filterSpec = {'*.jpg;*.jpeg;*.png;*.tif;*.tiff', 'Retinal Images (*.jpg, *.jpeg, *.png, *.tif, *.tiff)'};
            [file, path] = uigetfile(filterSpec, 'Select Retinal Fundus Image');
            if isequal(file, 0), return; end

            fullPath = fullfile(path, file);
            app.loadImageFromFile(fullPath);
            app.currentDemoCase = [];
        end

        % ---- Load demo case -----------------------------------------
        function onLoadDemo(app)
            sel = app.DemoDropdown.Value;
            if strcmp(sel, 'Select demo case...'), return; end

            % Find matching case
            caseIdx = find(strcmp(sel, cellfun(@(c) c.label, app.demoCases, 'UniformOutput', false)));
            if isempty(caseIdx), return; end

            dc = app.demoCases{caseIdx};
            app.currentDemoCase = dc;

            % Load real image if available, otherwise synthesize
            if isfield(dc, 'imageFile') && ~isempty(dc.imageFile) && isfile(dc.imageFile)
                img = imread(dc.imageFile);
                if size(img, 3) == 4, img = img(:,:,1:3); end
                if ndims(img) == 2, img = cat(3, img, img, img); end
            else
                img = generateSyntheticFundus(dc.imageSynth, [512 512]);
            end
            app.currentImage    = img;
            app.currentEnhanced = img;
            app.currentScreening = [];
            app.showingEnhanced  = false;
            app.currentOverlay   = 'original';

            % Update exam header
            app.patientCounter = app.patientCounter + 1;
            app.examCounter    = app.examCounter + 1;
            pid = sprintf('DEMO-P%04d', app.patientCounter);
            eid = sprintf('DEMO-E%04d', app.examCounter);
            app.LblPatientID.Text = pid;
            app.LblExamID.Text    = eid;
            app.LblDateTime.Text  = datestr(now, 'dd-mmm-yyyy HH:MM');
            app.LblEye.Text       = 'Unknown';
            app.LblStatus.Text    = 'Image loaded — Demo';
            app.LblStatus.FontColor = app.CLR_TEXT_SEC;

            % Display image
            app.displayImage(img);
            app.resetResultPanels();

            % If ungradable demo case, immediately show quality info
            if dc.drGrade == -1
                app.populateQuality(dc.quality);
            end
        end

        % ---- Load image from file path ------------------------------
        function loadImageFromFile(app, fullPath)
            try
                img = imread(fullPath);
                if ndims(img) == 2
                    img = cat(3, img, img, img);  % grayscale → RGB
                end
                if size(img,3) == 4
                    img = img(:,:,1:3);  % drop alpha
                end
                app.currentImage    = img;
                app.currentEnhanced = img;
                app.currentScreening = [];
                app.showingEnhanced  = false;
                app.currentOverlay   = 'original';
                app.patientCounter   = app.patientCounter + 1;
                app.examCounter      = app.examCounter + 1;
                pid = sprintf('PT-%04d', app.patientCounter);
                eid = sprintf('EX-%04d', app.examCounter);
                app.LblPatientID.Text = pid;
                app.LblExamID.Text    = eid;
                app.LblDateTime.Text  = datestr(now,'dd-mmm-yyyy HH:MM');
                app.LblEye.Text       = 'Unknown';
                app.LblStatus.Text    = 'Image loaded';
                app.LblStatus.FontColor = app.CLR_TEXT_SEC;
                app.displayImage(img);
                app.resetResultPanels();
            catch ME
                uialert(app.UIFigure, ...
                    sprintf('Cannot open image: %s\n\nSupported formats: JPG, PNG, TIF.', ME.message), ...
                    'Image Error', 'Icon', 'warning');
            end
        end

        % ---- Analyze ------------------------------------------------
        function onAnalyze(app)
            if isempty(app.currentImage)
                uialert(app.UIFigure, 'Please upload a retinal image or load a demo case first.', ...
                    'No Image', 'Icon', 'info');
                return;
            end

            % If we have a demo case with pre-computed results, use them
            if ~isempty(app.currentDemoCase) && app.currentDemoCase.drGrade ~= -2
                app.runDemoAnalysis();
            else
                app.runRealAnalysis();
            end
        end

        % ---- Demo analysis (pre-computed results) -------------------
        function runDemoAnalysis(app)
            dc = app.currentDemoCase;
            stages = {'Checking image quality...', 'Enhancing image...', ...
                      'Detecting retinal structures...', 'Detecting lesions...', ...
                      'Estimating DR severity...', 'Finalizing result...'};
            weights = [15 30 45 60 80 100];

            app.PanelProgress.Visible = 'on';
            app.LblStatus.Text = 'Processing...';
            app.LblStatus.FontColor = app.CLR_ACCENT;

            for i = 1:numel(stages)
                app.LblProgressStage.Text = stages{i};
                app.ProgressBar.Value = weights(i);
                drawnow;
                pause(0.35);
            end

            % Enhance real image
            try
                [enhanced, ~] = enhanceRetinalImage(app.currentImage, app.cfg);
                app.currentEnhanced = enhanced;
            catch
                app.currentEnhanced = app.currentImage;
            end

            app.PanelProgress.Visible = 'off';
            app.ProgressBar.Value = 0;

            % Populate UI from demo case
            app.populateQuality(dc.quality);

            if dc.drGrade == -1
                % Ungradable
                app.LblStatus.Text     = 'Recapture Required';
                app.LblStatus.FontColor = app.CLR_DANGER;
                app.LblGradeNum.Text   = '✕';
                app.LblGradeNum.FontColor = app.CLR_DANGER;
                app.LblGradeName.Text  = 'IMAGE CANNOT BE GRADED';
                app.LblConfidence.Text = 'Poor image quality prevents reliable grading.';
                app.LblReferral.Text   = ['Reason: ' dc.quality.reason char(10) 'Action: ' dc.quality.recommendation];
                return;
            end

            % Populate findings
            app.populateFindings(dc.findings);
            app.populateStructures(dc.structures);
            app.populateGrading(dc.drGrade, dc.confidence, 'high');
            app.LblStatus.Text = 'Complete — Demo';
            app.LblStatus.FontColor = app.CLR_SUCCESS;
        end

        % ---- Real analysis using pipeline ---------------------------
        function runRealAnalysis(app)
            stages = {'Checking image quality...', 'Enhancing image...', ...
                      'Detecting retinal structures...', 'Detecting lesions...', ...
                      'Estimating DR severity...', 'Finalizing result...'};
            weights = [15 30 45 60 80 100];

            app.PanelProgress.Visible = 'on';
            app.LblStatus.Text = 'Processing...';
            app.LblStatus.FontColor = app.CLR_ACCENT;
            drawnow;

            for i = 1:numel(stages)
                app.LblProgressStage.Text = stages{i};
                app.ProgressBar.Value = weights(i);
                drawnow;
                pause(0.1);
            end

            try
                result = runScreeningPipeline(app.currentImage, app.cfg);
                app.currentScreening = result;
                app.currentEnhanced  = result.enhanced;

                app.PanelProgress.Visible = 'off';
                app.ProgressBar.Value = 0;

                app.populateQuality(result.quality);

                if ~isempty(result.error) && ~result.quality.gradable
                    app.LblStatus.Text = 'Recapture Required';
                    app.LblStatus.FontColor = app.CLR_DANGER;
                    app.LblGradeNum.Text = '✕';
                    app.LblGradeNum.FontColor = app.CLR_DANGER;
                    app.LblGradeName.Text = 'IMAGE CANNOT BE GRADED';
                    app.LblConfidence.Text = result.quality.reason;
                    app.LblReferral.Text   = result.quality.recommendation;
                    return;
                end

                if ~isempty(result.lesions)
                    app.populateFindings(result.lesions);
                end
                if ~isempty(result.structures)
                    app.populateStructures(result.structures);
                end
                if ~isempty(result.grading)
                    app.populateGrading(result.grading.grade, result.grading.confidence, result.grading.confidenceLevel);
                    app.LblReferral.Text = result.grading.referralText;
                end

                app.LblStatus.Text = 'Complete';
                app.LblStatus.FontColor = app.CLR_SUCCESS;

            catch ME
                app.PanelProgress.Visible = 'off';
                uialert(app.UIFigure, ...
                    sprintf('Analysis error: %s\nCheck that the image is a valid retinal fundus image.', ME.message), ...
                    'Analysis Failed', 'Icon', 'error');
                app.LblStatus.Text = 'Analysis failed';
                app.LblStatus.FontColor = app.CLR_DANGER;
            end
        end

        % ---- Image display ------------------------------------------
        function displayImage(app, img)
            cla(app.AxImage);
            imshow(img, 'Parent', app.AxImage);
            app.AxImage.XAxis.Visible = 'off';
            app.AxImage.YAxis.Visible = 'off';
        end

        function onSetOverlay(app, key)
            app.currentOverlay = key;
            switch key
                case 'original'
                    if ~isempty(app.currentImage)
                        app.displayImage(app.currentImage);
                    end
                case 'enhanced'
                    if ~isempty(app.currentEnhanced)
                        app.displayImage(app.currentEnhanced);
                    end
                case 'vessel'
                    if ~isempty(app.currentScreening) && ~isempty(app.currentScreening.structures)
                        if isfield(app.currentScreening.structures.vessels, 'overlay')
                            app.displayImage(app.currentScreening.structures.vessels.overlay);
                        else
                            app.displayImage(app.currentEnhanced);
                        end
                    elseif ~isempty(app.currentImage)
                        app.displayImage(app.currentEnhanced);
                    end
                case 'lesion'
                    if ~isempty(app.currentScreening) && ~isempty(app.currentScreening.lesions)
                        app.displayImage(app.currentScreening.lesions.heatmap);
                    elseif ~isempty(app.currentEnhanced)
                        app.displayImage(app.currentEnhanced);
                    end
                case 'heatmap'
                    if ~isempty(app.currentScreening) && ~isempty(app.currentScreening.grading)
                        app.displayImage(app.currentScreening.grading.heatmap);
                    elseif ~isempty(app.currentEnhanced)
                        app.displayImage(app.currentEnhanced);
                    end
            end
        end

        % ---- Populate quality panel ---------------------------------
        function populateQuality(app, q)
            if isempty(q), return; end
            switch q.status
                case 'GOOD'
                    app.LblQualityStatus.Text      = 'GOOD';
                    app.LblQualityStatus.FontColor = app.CLR_SUCCESS;
                case 'BORDERLINE'
                    app.LblQualityStatus.Text      = 'BORDERLINE';
                    app.LblQualityStatus.FontColor = app.CLR_WARNING;
                otherwise
                    app.LblQualityStatus.Text      = 'UNGRADABLE';
                    app.LblQualityStatus.FontColor = app.CLR_DANGER;
            end
            app.LblQualityScore.Text = sprintf('%d / 100', q.score);
            app.LblFocus.Text        = q.focus;
            app.LblIllum.Text        = q.illumination;
            app.LblFOV.Text          = q.fieldOfView;
            app.LblContrast.Text     = q.contrast;
        end

        % ---- Populate findings panel --------------------------------
        function populateFindings(app, l)
            if isempty(l), return; end
            app.LblMA.Text       = app.formatFinding(l.microaneurysm.detected,    l.microaneurysm.confidence);
            app.LblExudate.Text  = app.formatFinding(l.exudate.detected,          l.exudate.confidence);
            app.LblHemorrhage.Text = app.formatFinding(l.hemorrhage.detected,     l.hemorrhage.confidence);
            app.LblNV.Text       = app.formatFinding(l.neovascularization.detected, l.neovascularization.confidence);

            if l.microaneurysm.detected,  app.LblMA.FontColor = app.CLR_WARNING;
            else,                          app.LblMA.FontColor = app.CLR_TEXT_PRI; end
            if l.exudate.detected,         app.LblExudate.FontColor = app.CLR_WARNING;
            else,                          app.LblExudate.FontColor = app.CLR_TEXT_PRI; end
            if l.hemorrhage.detected,      app.LblHemorrhage.FontColor = app.CLR_DANGER;
            else,                          app.LblHemorrhage.FontColor = app.CLR_TEXT_PRI; end
            if l.neovascularization.detected, app.LblNV.FontColor = app.CLR_DANGER;
            else,                          app.LblNV.FontColor = app.CLR_TEXT_PRI; end
        end

        % ---- Populate structures panel ------------------------------
        function populateStructures(app, s)
            if isempty(s), return; end
            app.LblOD.Text     = app.formatDetection(s.opticDisc.detected, s.opticDisc.confidence);
            app.LblFovea.Text  = app.formatDetection(s.fovea.detected,     s.fovea.confidence);
            app.LblVessels.Text = app.formatDetection(s.vessels.detected,  s.vessels.confidence);
        end

        % ---- Populate grading panel ---------------------------------
        function populateGrading(app, grade, confidence, confLevel)
            cfg_ = app.cfg;
            gradeColors = { ...
                app.CLR_SUCCESS, ...  % 0 No DR
                app.CLR_WARNING, ...  % 1 Mild
                app.CLR_WARNING, ...  % 2 Moderate
                app.CLR_DANGER,  ...  % 3 Severe
                app.CLR_PDR};         % 4 PDR

            if grade >= 0 && grade <= 4
                clr = gradeColors{grade+1};
            else
                clr = app.CLR_TEXT_SEC;
            end

            app.LblGradeNum.Text      = sprintf('Level %d', grade);
            app.LblGradeNum.FontColor = clr;
            app.LblGradeName.Text     = cfg_.dr.labels{grade+1};
            app.LblGradeName.FontColor = clr;
            app.LblConfidence.Text    = sprintf('Model confidence: %.0f%%  [%s]', confidence*100, upper(confLevel));

            if grade >= cfg_.dr.referralThreshold
                app.LblReferral.FontColor = app.CLR_DANGER;
            else
                app.LblReferral.FontColor = app.CLR_SUCCESS;
            end

            app.drawSeverityScale(grade);
        end

        % ---- Draw severity scale visualization ----------------------
        function drawSeverityScale(app, activeGrade)
            ax = app.AxSeverity;
            cla(ax);
            hold(ax, 'on');
            labels = {'No DR','Mild','Moderate','Severe','PDR'};
            gradeClrs = { ...
                [0.176 0.478 0.176], ...
                [0.769 0.490 0.000], ...
                [0.769 0.490 0.000], ...
                [0.710 0.188 0.000], ...
                [0.545 0.000 0.000]};
            for i = 1:5
                if i-1 == activeGrade
                    fc = gradeClrs{i};
                    tc = [1 1 1];
                    lw = 2;
                else
                    fc = [0.93 0.945 0.96];
                    tc = [0.60 0.65 0.70];
                    lw = 0.5;
                end
                rectangle(ax, 'Position', [(i-1)*0.2+0.005 0.05 0.19 0.90], ...
                    'FaceColor', fc, 'EdgeColor', [0.80 0.83 0.87], 'LineWidth', lw);
                text(ax, (i-1)*0.2+0.10, 0.65, labels{i}, ...
                    'HorizontalAlignment','center','FontSize',8,'FontWeight','bold','Color',tc);
                text(ax, (i-1)*0.2+0.10, 0.28, sprintf('%d',i-1), ...
                    'HorizontalAlignment','center','FontSize',10,'FontWeight','bold','Color',tc);
            end
            ax.XLim = [0 1]; ax.YLim = [0 1];
            ax.XAxis.Visible = 'off'; ax.YAxis.Visible = 'off';
            hold(ax, 'off');
        end

        % ---- Generate report ----------------------------------------
        function onGenerateReport(app)
            if isempty(app.currentImage)
                uialert(app.UIFigure, 'Please run an analysis first.', 'No Analysis', 'Icon', 'info');
                return;
            end

            try
                patientInfo.patientID     = app.LblPatientID.Text;
                patientInfo.examinationID = app.LblExamID.Text;
                patientInfo.eye           = app.LblEye.Text;
                patientInfo.date          = app.LblDateTime.Text;
                patientInfo.imageSource   = 'Uploaded Image';

                % If demo case, build a synthetic screening struct
                if ~isempty(app.currentDemoCase) && isempty(app.currentScreening)
                    dc = app.currentDemoCase;
                    screening.quality    = dc.quality;
                    screening.enhanced   = app.currentEnhanced;
                    screening.enhanceSteps = struct();
                    screening.structures = dc.structures;
                    screening.lesions    = dc.findings;
                    if dc.drGrade >= 0
                        screening.grading.grade = dc.drGrade;
                        screening.grading.confidence = dc.confidence;
                        screening.grading.confidenceLevel = 'moderate';
                        screening.grading.label = app.cfg.dr.labels{dc.drGrade+1};
                        screening.grading.shortLabel = app.cfg.dr.shortLabels{dc.drGrade+1};
                        cfg_ = app.cfg;
                        if dc.drGrade == 0
                            screening.grading.referralText = 'Routine follow-up. Rescreen in 12 months.';
                        elseif dc.drGrade <= 1
                            screening.grading.referralText = 'Rescreen in 6 months. Optimize glycemic control.';
                        else
                            screening.grading.referralText = 'Ophthalmology referral recommended.';
                        end
                        screening.grading.demoMode = true;
                    else
                        screening.grading = [];
                    end
                    screening.error = '';
                else
                    screening = app.currentScreening;
                    if isempty(screening)
                        uialert(app.UIFigure,'Please analyze the image before generating a report.','No Results','Icon','info');
                        return;
                    end
                end

                reportPath = generateReport(screening, patientInfo, app.cfg);
                app.reportPaths{end+1} = reportPath;

                % Open in browser
                web(reportPath, '-browser');

                uialert(app.UIFigure, ...
                    sprintf('Report saved:\n%s\n\nOpening in browser...', reportPath), ...
                    'Report Generated', 'Icon', 'success');
            catch ME
                uialert(app.UIFigure, ...
                    sprintf('Report generation error: %s', ME.message), ...
                    'Report Error', 'Icon', 'error');
            end
        end

        % ---- New examination ----------------------------------------
        function onNewCase(app)
            app.currentImage    = [];
            app.currentEnhanced = [];
            app.currentScreening = [];
            app.currentDemoCase = [];
            app.showingEnhanced  = false;
            app.currentOverlay   = 'original';
            cla(app.AxImage);
            text(app.AxImage, 0.5, 0.5, ...
                {'Upload a retinal fundus image', 'or load a demo case to begin'}, ...
                'Units','normalized','HorizontalAlignment','center', ...
                'FontSize', 13, 'Color', [0.5 0.55 0.6]);
            app.LblPatientID.Text   = '—';
            app.LblExamID.Text      = '—';
            app.LblDateTime.Text    = datestr(now,'dd-mmm-yyyy HH:MM');
            app.LblEye.Text         = 'Unknown';
            app.LblStatus.Text      = 'Not Analyzed';
            app.LblStatus.FontColor = app.CLR_TEXT_SEC;
            app.resetResultPanels();
        end

        % ---- Reset result panels ------------------------------------
        function resetResultPanels(app)
            app.LblQualityStatus.Text = '—';
            app.LblQualityStatus.FontColor = app.CLR_TEXT_SEC;
            app.LblQualityScore.Text = '— / 100';
            app.LblFocus.Text   = '—';
            app.LblIllum.Text   = '—';
            app.LblFOV.Text     = '—';
            app.LblContrast.Text = '—';
            app.LblMA.Text       = '—'; app.LblMA.FontColor = app.CLR_TEXT_PRI;
            app.LblExudate.Text  = '—'; app.LblExudate.FontColor = app.CLR_TEXT_PRI;
            app.LblHemorrhage.Text = '—'; app.LblHemorrhage.FontColor = app.CLR_TEXT_PRI;
            app.LblNV.Text       = '—'; app.LblNV.FontColor = app.CLR_TEXT_PRI;
            app.LblOD.Text       = '—';
            app.LblFovea.Text    = '—';
            app.LblVessels.Text  = '—';
            app.LblGradeNum.Text = '—';
            app.LblGradeNum.FontColor = app.CLR_TEXT_SEC;
            app.LblGradeName.Text = 'Not analyzed';
            app.LblGradeName.FontColor = app.CLR_TEXT_PRI;
            app.LblConfidence.Text = 'Model confidence: —';
            app.LblReferral.Text = 'Upload and analyze a retinal image to begin screening.';
            app.LblReferral.FontColor = app.CLR_TEXT_PRI;
            app.drawSeverityScale(-1);
        end

        % ---- Text formatters ----------------------------------------
        function s = formatFinding(~, detected, conf)
            if detected
                s = sprintf('Detected  (%.0f%%)', conf*100);
            else
                s = sprintf('Not detected  (%.0f%%)', conf*100);
            end
        end

        function s = formatDetection(~, detected, conf)
            if detected
                s = sprintf('Detected — conf. %.0f%%', conf*100);
            else
                s = sprintf('Not detected');
            end
        end
    end
end
