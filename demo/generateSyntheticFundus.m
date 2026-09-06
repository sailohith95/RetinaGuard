function img = generateSyntheticFundus(params, targetSize)
% generateSyntheticFundus  Creates a synthetic retinal fundus-like image.
%
%   img = generateSyntheticFundus(params, targetSize)
%
%   params  - struct with fields:
%               .lesionCount  - number of simulated lesion spots
%               .noiseLevel   - Gaussian noise sigma (0-1)
%               .illumination - 'normal' | 'slightly_dark' | 'dark'
%   targetSize - [rows cols] e.g. [512 512]
%
%   Returns: img (uint8 RGB H×W×3)
%
%   DEMO NOTE: This image is synthetically generated for demonstration
%   purposes only. It does NOT represent a real retinal examination.

if nargin < 2, targetSize = [512 512]; end

rows = targetSize(1);
cols = targetSize(2);
cx   = cols / 2;
cy   = rows / 2;
R    = min(rows, cols) * 0.46;  % retinal disc radius

% --- Base: dark background ---
img = zeros(rows, cols, 3, 'double');

% --- Create circular retinal field ---
[X, Y] = meshgrid(1:cols, 1:rows);
dist   = sqrt((X - cx).^2 + (Y - cy).^2);
mask   = dist <= R;

% --- Warm reddish-orange fundus base tone ---
baseR = 0.55 * mask;
baseG = 0.22 * mask;
baseB = 0.08 * mask;

% --- Vignette (brighter centre, darker edges) ---
vignette = exp(-((dist / R).^2) * 0.6);
baseR = baseR .* (0.7 + 0.3 * vignette);
baseG = baseG .* (0.7 + 0.3 * vignette);
baseB = baseB .* (0.7 + 0.3 * vignette);

% --- Blood vessel pattern (low-level) ---
% Simulate vessels as thin dark arcs
for angle = linspace(0, 2*pi, 12)
    vx = cx + linspace(0, R*0.85, 200) .* cos(angle);
    vy = cy + linspace(0, R*0.85, 200) .* sin(angle);
    for k = 1:length(vx)
        px = round(vx(k));
        py = round(vy(k));
        if px >= 1 && px <= cols && py >= 1 && py <= rows
            % Thin dark line
            for dr = -1:1
                for dc = -1:1
                    nr = py + dr; nc = px + dc;
                    if nr>=1 && nr<=rows && nc>=1 && nc<=cols && mask(nr,nc)
                        baseR(nr,nc) = baseR(nr,nc) * 0.5;
                        baseG(nr,nc) = baseG(nr,nc) * 0.5;
                        baseB(nr,nc) = baseB(nr,nc) * 0.4;
                    end
                end
            end
        end
    end
end

% --- Optic disc (bright yellowish region) ---
odX = cx + R * 0.35;
odY = cy - R * 0.02;
odR = R * 0.12;
odMask = sqrt((X - odX).^2 + (Y - odY).^2) <= odR;
baseR(odMask) = 0.92;
baseG(odMask) = 0.85;
baseB(odMask) = 0.55;

% --- Fovea (small dark depression) ---
fvX = cx - R * 0.12;
fvY = cy;
fvR = R * 0.05;
fvMask = sqrt((X - fvX).^2 + (Y - fvY).^2) <= fvR;
baseR(fvMask) = baseR(fvMask) * 0.55;
baseG(fvMask) = baseG(fvMask) * 0.55;
baseB(fvMask) = baseB(fvMask) * 0.55;

% --- Simulate lesions ---
rng(42); % reproducible
lesionCount = params.lesionCount;
for i = 1:lesionCount
    theta  = rand() * 2 * pi;
    radius = rand() * R * 0.75 + 0.05 * R;
    lx     = round(cx + radius * cos(theta));
    ly     = round(cy + radius * sin(theta));
    lSize  = randi([3 8]);
    lType  = mod(i, 3);  % 0=MA(dark red), 1=exudate(bright), 2=hemorr(dark)

    for dr = -lSize:lSize
        for dc = -lSize:lSize
            if dr^2 + dc^2 <= lSize^2
                nr = ly + dr; nc = lx + dc;
                if nr>=1&&nr<=rows&&nc>=1&&nc<=cols&&mask(nr,nc)
                    if lType == 0        % Microaneurysm – small dark red dot
                        baseR(nr,nc) = 0.45;
                        baseG(nr,nc) = 0.05;
                        baseB(nr,nc) = 0.05;
                    elseif lType == 1    % Hard exudate – bright yellowish
                        baseR(nr,nc) = 0.90;
                        baseG(nr,nc) = 0.85;
                        baseB(nr,nc) = 0.50;
                    else                 % Hemorrhage – dark red blotch
                        baseR(nr,nc) = 0.30;
                        baseG(nr,nc) = 0.01;
                        baseB(nr,nc) = 0.01;
                    end
                end
            end
        end
    end
end

% --- Illumination modulation ---
switch params.illumination
    case 'slightly_dark'
        scale = 0.72;
    case 'dark'
        scale = 0.35;
    otherwise
        scale = 1.0;
end
baseR = baseR * scale;
baseG = baseG * scale;
baseB = baseB * scale;

% --- Gaussian noise ---
noise = params.noiseLevel;
baseR = baseR + noise * randn(rows, cols);
baseG = baseG + noise * randn(rows, cols);
baseB = baseB + noise * randn(rows, cols);

% --- Clamp and convert ---
img(:,:,1) = baseR;
img(:,:,2) = baseG;
img(:,:,3) = baseB;
img = max(0, min(1, img));
img = im2uint8(img);
end
