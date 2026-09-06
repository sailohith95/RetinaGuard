%% RetinaGuard — Test Suite
%
%   Run this script to validate core functions with synthetic images.
%   No real dataset required.
%
%   Usage:
%       run('RetinaGuard/runAllTests.m')

fprintf('\n');
fprintf('=================================================================\n');
fprintf('  RetinaGuard Test Suite\n');
fprintf('=================================================================\n\n');

thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(thisDir));

cfg = RGConfig();
passed = 0; failed = 0; total = 0;

%% Test 1: Config loads
total = total + 1;
try
    assert(~isempty(cfg.app.name));
    assert(~isempty(cfg.dr.labels));
    fprintf('[PASS] Config loaded: %s v%s\n', cfg.app.name, cfg.app.version);
    passed = passed + 1;
catch e
    fprintf('[FAIL] Config: %s\n', e.message);
    failed = failed + 1;
end

%% Test 2: Synthetic fundus generation
total = total + 1;
try
    params.lesionCount = 10;
    params.noiseLevel  = 0.03;
    params.illumination = 'normal';
    img = generateSyntheticFundus(params, [256 256]);
    assert(isa(img,'uint8'));
    assert(size(img,1)==256 && size(img,2)==256 && size(img,3)==3);
    fprintf('[PASS] Synthetic fundus generation: %dx%dx%d uint8\n', size(img,1), size(img,2), size(img,3));
    passed = passed + 1;
catch e
    fprintf('[FAIL] Synthetic fundus: %s\n', e.message);
    failed = failed + 1;
end

%% Test 3: Image quality assessment
total = total + 1;
try
    params.lesionCount = 0; params.noiseLevel = 0.02; params.illumination = 'normal';
    img = generateSyntheticFundus(params, [256 256]);
    q = assessImageQuality(img, cfg);
    assert(isfield(q,'score'));
    assert(isfield(q,'status'));
    assert(isfield(q,'gradable'));
    assert(q.score >= 0 && q.score <= 100);
    fprintf('[PASS] Quality assessment: score=%d, status=%s, gradable=%d\n', q.score, q.status, q.gradable);
    passed = passed + 1;
catch e
    fprintf('[FAIL] Quality assessment: %s\n', e.message);
    failed = failed + 1;
end

%% Test 4: Ungradable image detection
total = total + 1;
try
    params.lesionCount = 0; params.noiseLevel = 0.40; params.illumination = 'dark';
    imgBad = generateSyntheticFundus(params, [256 256]);
    q2 = assessImageQuality(imgBad, cfg);
    assert(~q2.gradable || q2.score < 50, 'Expected low quality for dark blurry image');
    fprintf('[PASS] Ungradable detection: score=%d, gradable=%d\n', q2.score, q2.gradable);
    passed = passed + 1;
catch e
    fprintf('[FAIL] Ungradable detection: %s\n', e.message);
    failed = failed + 1;
end

%% Test 5: Image enhancement
total = total + 1;
try
    params.lesionCount = 0; params.noiseLevel = 0.02; params.illumination = 'normal';
    img = generateSyntheticFundus(params, [256 256]);
    [enhanced, steps] = enhanceRetinalImage(img, cfg);
    assert(isa(enhanced,'uint8'));
    assert(isfield(steps,'enhanced'));
    fprintf('[PASS] Image enhancement: output size %dx%d\n', size(enhanced,1), size(enhanced,2));
    passed = passed + 1;
catch e
    fprintf('[FAIL] Image enhancement: %s\n', e.message);
    failed = failed + 1;
end

%% Test 6: Structure detection
total = total + 1;
try
    params.lesionCount = 0; params.noiseLevel = 0.02; params.illumination = 'normal';
    img = generateSyntheticFundus(params, [512 512]);
    [enhanced, ~] = enhanceRetinalImage(img, cfg);
    s = detectStructures(enhanced, cfg);
    assert(isfield(s,'opticDisc'));
    assert(isfield(s,'fovea'));
    assert(isfield(s,'vessels'));
    assert(isfield(s.opticDisc, 'mask'), 'Expected optic disc mask');
    assert(isfield(s.opticDisc, 'radius'), 'Expected optic disc radius');
    assert(isfield(s.vessels, 'density'), 'Expected vessel density');
    assert(s.demoMode == true);
    fprintf('[PASS] Structure detection: OD=%d (%.0f%%), Fovea=%d (%.0f%%), VesselDensity=%.2f%%\n', ...
        s.opticDisc.detected, s.opticDisc.confidence*100, ...
        s.fovea.detected, s.fovea.confidence*100, s.vessels.density);
    passed = passed + 1;
catch e
    fprintf('[FAIL] Structure detection: %s\n', e.message);
    failed = failed + 1;
end

%% Test 7: Lesion detection
total = total + 1;
try
    params.lesionCount = 20; params.noiseLevel = 0.03; params.illumination = 'normal';
    img = generateSyntheticFundus(params, [512 512]);
    [enhanced, ~] = enhanceRetinalImage(img, cfg);
    s = detectStructures(enhanced, cfg);
    l = detectLesions(enhanced, cfg, s);
    assert(isfield(l,'microaneurysm'));
    assert(isfield(l,'exudate'));
    assert(isfield(l,'hemorrhage'));
    assert(isfield(l,'neovascularization'));
    assert(isfield(l.neovascularization, 'riskLevel'), 'Expected neovascularization risk level');
    assert(isfield(l.exudate, 'area'), 'Expected exudate area');
    assert(isfield(l.hemorrhage, 'area'), 'Expected hemorrhage area');
    assert(l.demoMode == true);
    fprintf('[PASS] Lesion detection: MA=%d, Exudate=%d (area=%d), Hemorrhage=%d (area=%d), NV Risk=%s\n', ...
        l.microaneurysm.detected, l.exudate.detected, l.exudate.area, ...
        l.hemorrhage.detected, l.hemorrhage.area, l.neovascularization.riskLevel);
    passed = passed + 1;
catch e
    fprintf('[FAIL] Lesion detection: %s\n', e.message);
    failed = failed + 1;
end

%% Test 8: DR grading (placeholder)
total = total + 1;
try
    params.lesionCount = 15; params.noiseLevel = 0.03; params.illumination = 'normal';
    img = generateSyntheticFundus(params, [512 512]);
    [enhanced, ~] = enhanceRetinalImage(img, cfg);
    q = assessImageQuality(enhanced, cfg);
    l = detectLesions(enhanced, cfg);
    g = gradeDR(enhanced, l, q, cfg);
    assert(isfield(g,'grade'));
    assert(g.grade >= 0 && g.grade <= 4);
    assert(isfield(g,'confidence'));
    assert(islogical(g.demoMode));
    fprintf('[PASS] DR grading: grade=%d (%s), conf=%.0f%%, demoMode=%d\n', g.grade, g.label, g.confidence*100, g.demoMode);
    passed = passed + 1;
catch e
    fprintf('[FAIL] DR grading: %s\n', e.message);
    failed = failed + 1;
end

%% Test 9: Full pipeline
total = total + 1;
try
    params.lesionCount = 8; params.noiseLevel = 0.03; params.illumination = 'normal';
    img = generateSyntheticFundus(params, [512 512]);
    result = runScreeningPipeline(img, cfg);
    assert(isfield(result,'quality'));
    assert(isfield(result,'grading'));
    fprintf('[PASS] Full pipeline: grade=%d, quality=%s\n', ...
        result.grading.grade, result.quality.status);
    passed = passed + 1;
catch e
    fprintf('[FAIL] Full pipeline: %s\n', e.message);
    failed = failed + 1;
end

%% Test 10: Demo cases
total = total + 1;
try
    cases = getDemoCases();
    assert(numel(cases) == 6);
    for k = 1:6
        assert(isfield(cases{k},'drGrade'));
        assert(isfield(cases{k},'quality'));
    end
    fprintf('[PASS] Demo cases: %d cases loaded\n', numel(cases));
    passed = passed + 1;
catch e
    fprintf('[FAIL] Demo cases: %s\n', e.message);
    failed = failed + 1;
end

%% Test 11: Report generation
total = total + 1;
try
    params.lesionCount = 8; params.noiseLevel = 0.03; params.illumination = 'normal';
    img = generateSyntheticFundus(params, [512 512]);
    screening = runScreeningPipeline(img, cfg);
    patientInfo.patientID = 'TEST-001';
    patientInfo.examinationID = 'EXAM-001';
    patientInfo.eye = 'Right';
    patientInfo.date = datestr(now, 'dd-mmm-yyyy HH:MM');
    rPath = generateReport(screening, patientInfo, cfg);
    assert(isfile(rPath));
    fprintf('[PASS] Report generated: %s\n', rPath);
    passed = passed + 1;
catch e
    fprintf('[FAIL] Report generation: %s\n', e.message);
    failed = failed + 1;
end

%% Summary
fprintf('\n=================================================================\n');
fprintf('  Results: %d/%d passed  |  %d failed\n', passed, total, failed);
if failed == 0
    fprintf('  ALL TESTS PASSED\n');
else
    fprintf('  %d test(s) failed — review above messages\n', failed);
end
fprintf('=================================================================\n\n');
