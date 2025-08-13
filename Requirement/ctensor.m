function C = ctensor(dim1, dim2, dim3, type, varargin)
    %CTENSOR 生成dim1×dim2的矩阵集合，存储在长度dim3的cell数组中
    % 参数与之前tensor函数类似

    if any([dim1, dim2, dim3] <= 0) || any(fix([dim1, dim2, dim3]) ~= [dim1, dim2, dim3])
        error('维度参数必须是正整数。');
    end

    idxSeed = find(strcmpi(varargin, 'seed'), 1);
    if ~isempty(idxSeed) && idxSeed < numel(varargin)
        rng(varargin{idxSeed + 1});
        varargin([idxSeed, idxSeed + 1]) = [];
    end

    C = cell(1, dim3);

    switch lower(type)
        case 'ones'
            for k = 1:dim3
                C{k} = ones(dim1, dim2);
            end
        case 'zeros'
            for k = 1:dim3
                C{k} = zeros(dim1, dim2);
            end
        case 'rand'
            for k = 1:dim3
                C{k} = rand(dim1, dim2);
            end
        case 'randn'
            for k = 1:dim3
                C{k} = randn(dim1, dim2);
            end
        case 'const'
            if isempty(varargin)
                error('''const'' 类型需要提供常数值。');
            end
            val = varargin{1};
            for k = 1:dim3
                C{k} = val * ones(dim1, dim2);
            end
        case 'orth'
            for k = 1:dim3
                A = randn(dim1, dim2);
                if dim1 < dim2
                    [Q, ~] = qr(A', 'econ');
                    Q = Q';
                else
                    [Q, ~] = qr(A, 'econ');
                end
                C{k} = Q;
            end
        case 'eye'
            I = eye(dim1, dim2);
            for k = 1:dim3
                C{k} = I;
            end
        otherwise
            error('不支持的类型：%s。', type);
    end
    if dim3 == 1 
        id = find(strcmpi(varargin, 'matirx'), 1);
        if ~isempty(id) && varargin{id + 1}
            C = C{1};
        end
    end
end
