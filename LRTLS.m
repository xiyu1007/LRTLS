classdef My
    properties
        % 基本参数
        name = 'My';
        max_iter = 200;
        tol = 1e-3;
        toleration = 1e-9;
        iter_tol = 1e-3;

        verbose = 0;
        keyboard = 0;
        Loss = [];
        Err = [];

        start_time=1;
        runtime = 0;
        
        % 超参数
        alpha;   
        beta;    
        lambda;  
        gamma;
        eta;
        mu;
        h;
        t;
        
        % 数据维度
        n;   % 样本数
        M;     % 视图数
        d;        % 各视图特征维度[]
        c;  
        
        % 优化变量
        W;     
        A;
        Z;
        D;
        Q;
        F;  

        G;
        R
      
        rho = 1;
        rho_max = 1e6;
        delta = 1.1;
        epsilon = 1e-5;

    end
    
    methods
        function obj = My()
            obj.Err = NaN(3,obj.max_iter);
        end

        % Set parameters
        function obj = setParams(obj, params)
            obj.alpha = params(1);
            obj.beta = params(2);
            obj.mu = params(3);
            % obj.lambda = params(4);
            obj.lambda = params(4) * params(1);
            obj.gamma = params(5);
            obj.eta = params(6);
            obj.h = 5;
            obj.t = obj.h;
            % obj.mu = 0;
        end

        % Initialize
        function obj = init(obj,n, c, M, d, seed)
            obj.M = M;
            obj.n = n;
            obj.d = d;
            obj.c = c; 
            obj.h = c * obj.h;
            obj.t = obj.h;
            h = obj.h;
            t = obj.t;

            rng(seed);
            obj.W = cell(1,obj.M);
            for m =1:M
              obj.W{m} = ctensor(d(m), h, 1, 'zeros', 'seed', seed,'matirx',1);
            end
            obj.D = ctensor(t, n, obj.M, 'zeros', 'seed', seed);
            obj.Z = ctensor(t, n, obj.M, 'randn', 'seed', seed);
            obj.Q = ctensor(t, n, 1, 'zeros', 'seed', seed,'matirx',1);
            obj.A = ctensor(h, t, 1, 'orth', 'seed', seed,'matirx',1);
            obj.F = ctensor(c, n, 1, 'orth', 'seed', seed,'matirx',1);

            % obj.G = obj.Z;
            obj.G = ctensor(t, n, obj.M, 'zeros',  'seed', seed);
            obj.R = ctensor(t, n, obj.M, 'zeros', 'seed', seed);

            obj.start_time = tic;
        end

        function W = updateW(obj,X)
            Z = obj.Z;
            W = obj.W;
            A = obj.A;
            D = obj.D;
            % L = obj.L;

            beta = obj.beta;
            alpha = obj.alpha;
            mu = obj.mu;

            for m = 1:obj.M
                Xm = X{m};
                norm2row = vecnorm(W{m}, 2, 2);
                [~, df] = Geman(norm2row, mu);
                Dw = diag(df ./ (2 * norm2row + obj.iter_tol));
                ep = obj.tol * eye(obj.d(m));
                term1 = alpha * Xm*(Xm') + beta*Dw + ep;
                term2 = alpha * Xm * ( A*(Z{m}+D{m}) )';
                W{m} = term1 \ term2;
            end
        end

        function Z = updateZ(obj,X,Y)
            Z = obj.Z;
            R = obj.R;
            Q = obj.Q;
            A = obj.A;
            eta = obj.eta;
            gamma = obj.gamma;
            G = obj.G;
            D = obj.D;
            W = obj.W;
            F = obj.F;
            rho = obj.rho;
            rho2 = rho / 2;
            alpha = obj.alpha;

            I = eye(obj.t);
            sumZ = 0;
            for m=1:obj.M
                sumZ = sumZ + Z{m};
            end

            for m=1:obj.M
                WTX = W{m}' * X{m};
                WTXD = WTX - A * D{m};
                sumZ = sumZ - Z{m};
                temp = obj.M * (F')*Y - Q'* sumZ;
                term1 = alpha * (A')*A + eta*Q*(Q') + (gamma + rho2) * I;
                term2 = alpha * (A') * WTXD + gamma*Q + eta*Q*temp + rho2* (G{m} + (R{m}/rho));
                Z{m} = term1 \ term2;
                sumZ = sumZ + Z{m};
            end
        end

        function A = updateA(obj,X)
            Z = obj.Z;
            W = obj.W;
            D = obj.D;

            C = 0;
            for m=1:obj.M
                C = C + ( W{m}'*X{m} ) * (Z{m}+D{m})';
            end
            % A = W';
            [U,~,V] = svd(C,'econ');
            A = U*V';
        end

        function F = updateF(obj,Y)
            Q = obj.Q;
            Z = obj.Z;

            C = 0;
            for m=1:obj.M
                C = C + (Q'*Z{m});
            end
            R = obj.M * C * Y';
            [U,~,V] = svd(R,"econ");
            F = (U*V')';  % 转置得到 F ∈ [c x k]
        end

        function D = updateD(obj,X)
            Z = obj.Z;
            D = obj.D;
            lambda = obj.lambda;
            A = obj.A;
            W = obj.W;
            I = eye(obj.t);
            alpha = obj.alpha;

            sumD = 0;
            for m=1:obj.M
                sumD = sumD + D{m};
            end
            
            for m=1:obj.M
                WXAZ  = W{m}'*X{m} - A*Z{m};
                sumD = sumD - D{m};
                term1 = alpha * (A')*A + lambda * I;
                term2 = alpha * A'* WXAZ - lambda* 0.5 *sumD;
                D{m} =  term1 \ term2;
                sumD = sumD + D{m};
            end
        end

        function Q = updateQ(obj,Y)
            F = obj.F;
            gamma = obj.gamma;
            Z = obj.Z;
            eta = obj.eta;
            I = eye(obj.t);
            sumZ = 0;
            for m=1:obj.M
                sumZ = sumZ + Z{m};
            end
            term1 = obj.M* gamma* I + eta * sumZ*(sumZ');
            term2 = eta* sumZ * (obj.M * (F')*Y)' + gamma* sumZ;
            Q = term1 \ term2;
        end

        function [G,fTnn] = updateG(obj)
            ref = 1 / obj.rho;
            Rt = cat(3, obj.Z{:}) - (cat(3, obj.R{:}) ./obj.rho);
            [G,fTnn] = TNN(Rt,ref);
        end

        function [R,rho] = updateLagrange(obj)
            G = obj.G;
            Z = obj.Z;
            delta = obj.delta;
            rho = obj.rho;
            R = obj.R;

            for m=1:obj.M
                R{m} = R{m} + rho*(G{m} - Z{m});
            end
            rho = min(obj.rho_max,delta*rho);
        end

        function [f,df] = Fun(obj, X, Y,fTnn)
            Z = obj.Z;
            W = obj.W;
            A = obj.A;
            G = obj.G;
            F = obj.F;
            D = obj.D;
            Q = obj.Q;
            R = obj.R;
            alpha = obj.alpha;
            lambda = obj.lambda;
            beta = obj.beta;
            gamma = obj.gamma;
            eta = obj.eta;
            mu = obj.mu;

            f1 = 0;
            f2 = 0;
            f3 = 0;
            if nargin > 3
                f3 = fTnn;
            end
            f4 = 0;
            f5 = 0; 
            f6 = 0;
            f8 = 0;
            % f9 = 0;
            fw = 0;
            rho = obj.rho;
            rho2 = rho / 2;
            sumZ = 0;
            for m=1:obj.M
                sumZ = sumZ + Z{m};
            end
            f6 = f6 + norm(Q'*sumZ - obj.M*(F')*Y,'fro')^2;
           
            for m=1:obj.M
                f1 = f1 + norm(W{m}'*X{m} - A*(Z{m}+D{m}),'fro')^2;
                norm2W = vecnorm(W{m},2,2);
                [wrow, ~] = Geman(norm2W, mu);
                f2 = f2 + sum(wrow);
                if nargin <= 3
                    [~,Sg,~] = svd(G{m},'econ');
                    f3 = f3 + sum(diag(Sg));
                end

                for u=m:obj.M
                    f4 = f4 + trace(D{m}'*D{u});
                end
                f5 = f5 + norm(Q - Z{m},'fro')^2;
                f8 = f8 + norm(G{m} - Z{m} + (R{m}/rho),'fro')^2;
                fw = fw + obj.tol*norm(W{m},'fro')^2;
            end
            f = alpha* f1 + beta*f2 + f3 + lambda*f4  + gamma*f5 + eta*f6 + rho2*(f8) + fw;
            % f = alpha* f1 + beta*f2 + f3 + lambda*f4  + gamma*f5 + eta*f6 + rho2*(f8) + mu*f9 + fw ;
            df = 0;
        end

        function obj = run(obj,X,Y,param,seed)
            warning('off');
            obj.verbose = 1;
            % obj.keyboard = 1;
            % profile on
            if obj.verbose
                warning('on');
            end

            rng(seed);
            [n, c, M, d] = getDataInfo(X,Y);
            max_iter = obj.max_iter;

            obj = setParams(obj, param);
            obj = obj.init(n, c, M, d, seed);

            [X, Y] = TransposeXY(X, Y);
            invD = diag(1 ./ sqrt(diag(Y*Y')));  % c x c
            obj.F = invD * Y;

            [fo,~] = obj.Fun(X,Y);
            fTnn = 0;
            for iter = 1:max_iter
                obj.W = obj.updateW(X);
                obj.A = obj.updateA(X);
                obj.Q = obj.updateQ(Y);   
                [obj.G,fTnn] = obj.updateG();
                obj.Z = obj.updateZ(X,Y);
                obj.D = obj.updateD(X);       
                obj.F = obj.updateF(Y);
                [obj.R,obj.rho] = obj.updateLagrange();

                C1 = 0;
                for m=1:M
                    C1 = max( C1, norm(obj.G{m} - obj.Z{m}, Inf) );
                end
                [f,~] = obj.Fun(X,Y,fTnn);
                obj.Loss(iter) = f;
                obj.Err(1,iter) = C1;
                rate = (fo - f)/fo;
                if C1 < obj.iter_tol && rate <  obj.iter_tol && iter >8
                    break;
                end
                fo = f;
            end

            if obj.verbose % || 1
                figure
                linew = 1.2;
                % 归一化处理
                norm_Loss = obj.Loss ./ max(obj.Loss);
                norm_Err = obj.Err ./ max(obj.Err, [], 2); % 每行分别归一化
                % 绘图
                plot(norm_Loss, 'LineWidth', linew); hold on;
                for i = 1:size(obj.Err,1)
                    plot(norm_Err(i,:), 'LineWidth', linew);
                end
                % 图例标签
                legend({...
                    '$\mathrm{Loss}$','$\|\mathcal{G} - \mathcal{Z}\|_\infty$'}, ...
                    'Interpreter', 'latex', 'FontSize', 11, 'Location', 'northeast');
            
                xlabel('Iteration', 'Interpreter', 'latex');
                ylabel('Normalized Value', 'Interpreter', 'latex');
                yticks(0:0.2:1);
                grid on;
                title('Loss and Convergence Conditions', 'Interpreter', 'latex');
                % fig = gcf;
                % exportgraphics(fig, ['Analyze\Fig\','Loss_A1.pdf'], 'ContentType', 'vector');
                % close;
            end
            % profile viewer
        end

        function parameter = init_param(~,fix)
            if nargin < 2
                fix = [Inf, Inf, Inf, Inf, Inf Inf, Inf, Inf, Inf];
            end
            
            alphaSpace = [0.01 0.1 0.5 1 5 10 100];
            betaSpace = [0.01 0.1 0.5 1 5 10 100];
            muSpace = [0.1 0.3 0.5 1 5 10];
            lambdaSpace = [0.001 0.01 0.1 0.5 1 5 10 50 100 1000];
            gammaSpace = [0.01 0.1 0.5 1 5 10 100];
            etaSpace = [0.1 1 10 50 100 1000];
            % hSpace = [5 10 20 30 40 50 100];

            if fix(1) ~= Inf
                alphaSpace = fix(1);
            end
            if fix(2) ~= Inf
                betaSpace = fix(2);
            end
            if fix(3) ~= Inf
                muSpace = fix(3);
            end
            if fix(4) ~= Inf
                lambdaSpace = fix(4);
            end
            if fix(5) ~= Inf
                gammaSpace = fix(5);
            end
            if fix(6) ~= Inf
                etaSpace = fix(6);
            end

            paramSpace = {alphaSpace, betaSpace, muSpace, lambdaSpace, gammaSpace, etaSpace};
            % paramSpace = {alphaSpace, betaSpace, lambdaSpace, gammaSpace, etaSpace,muSpace,hSpace};
            parameter = combvec(paramSpace{:})';
            parameter = sortrows(parameter, 'ascend');
        end

    end
end
