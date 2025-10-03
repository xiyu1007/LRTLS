function [f, df] = Geman(x, delta)
% GEMAN 计算 Geman 函数及其导数
% 输入:
%   x     - 输入向量或标量
%   delta - 正参数（通常为小正数）
% 输出:
%   f  - 函数值:    f(x) = delta * x / (x + delta)
%   df - 导数值:    f'(x) = delta^2 / (x + delta)^2
    x = abs(x);

    if nargin < 2
        % delta = 1e-3;  % 默认值
        delta = 1;  % 默认值
    end

    f = (delta .* x) ./ (x + delta);
    df = (delta^2) ./ (x + delta).^2;
end
