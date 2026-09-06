classdef DRClassifierModel
% DRClassifierModel  Interface wrapper for the DR severity classifier.
%
%   Usage:
%     model = DRClassifierModel(cfg);
%     result = model.predict(img);
%
%   When no trained network is available, returns demo inference flagged
%   with demoMode = true.
%
%   To connect a real MATLAB network:
%     1. Train or load a dlnetwork/SeriesNetwork/DAGNetwork.
%     2. Save it: save(cfg.paths.drModel, 'net', 'classes');
%     3. This class loads it automatically on the next predict() call.

    properties (Access = private)
        net       = []
        classes   = {}
        loaded    = false
        modelPath = ''
    end

    methods
        function obj = DRClassifierModel(cfg)
            if nargin < 1, cfg = RGConfig(); end
            obj.modelPath = cfg.paths.drModel;
            obj = obj.tryLoad();
        end

        function obj = tryLoad(obj)
            if isfile(obj.modelPath)
                try
                    data = load(obj.modelPath, 'net', 'classes');
                    obj.net     = data.net;
                    obj.classes = data.classes;
                    obj.loaded  = true;
                catch
                    obj.loaded = false;
                end
            end
        end

        function result = predict(obj, img)
        % predict  Returns DR grade prediction.
            result.demoMode = ~obj.loaded;

            if obj.loaded
                % --- Real network inference --------------------------
                try
                    cfg = RGConfig();
                    tSize = cfg.preprocess.targetSize;
                    imgR  = imresize(img, tSize);
                    scores = predict(obj.net, single(imgR));
                    [conf, gradeIdx] = max(scores);
                    result.grade        = gradeIdx - 1;  % 0-indexed
                    result.confidence   = double(conf);
                    result.probabilities = double(scores);
                    result.demoMode     = false;
                catch ME
                    result.grade = 0; result.confidence = 0;
                    result.probabilities = zeros(1,5);
                    result.error = ME.message;
                end
            else
                % --- Placeholder demo --------------------------------
                result.grade         = 0;
                result.confidence    = 0;
                result.probabilities = zeros(1,5);
                result.message       = 'Trained DR classifier not loaded. Demo mode active.';
            end
        end

        function tf = isLoaded(obj)
            tf = obj.loaded;
        end

        function status = getStatus(obj)
            if obj.loaded
                status = 'Trained Model Loaded';
            else
                status = 'Not Loaded — Demo Mode';
            end
        end
    end
end
