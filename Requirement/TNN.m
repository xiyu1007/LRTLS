function G = TNN(R,ref,fhandle,G)

        [~, ~, n3] = size(R);
        if isscalar(ref)
            ref = repmat(ref, 1, n3);
        end

        Rf = fft(R, [], 3); % t-SVD 是在频域上定义的，首先对张量 Q 做傅里叶变换：
        Gf = zeros(size(Rf));
        if nargin >= 4
            Gf = fft(G, [], 3);
        end

        for i = 1:n3
            [U, S, V] = svd(Rf(:,:,i), 'econ');
            s = diag(S);
            if nargin >= 3
                [~, Sg, ~] = svd(Gf(:,:,i), 'econ');
                sg = diag(Sg);
                [~,sg] = fhandle(sg); 
                new_vals = max(s - ref(i)*sg, 0);
            else
                new_vals = max(s - ref(i), 0);
            end
            S_new = diag(new_vals);
            Gf(:,:,i) = U * S_new * V'; % 在频域计算后再整体 ifft
        end
        G = ifft(Gf, [], 3);
        G = squeeze(num2cell(G, [1 2]));
end