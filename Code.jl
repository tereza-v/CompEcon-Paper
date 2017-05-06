
  using DataFrames, FastGaussQuadrature, Plots, Distributions

  # simulate a data set
    function simulate(n,A,theta,e16)
        # parameters: theta = return to educ, to exp, exp^2, cost of educ, cost of going back to school (removed from analysis to simplify)
        theta = theta

        e16 = e16
        g16 = 10  # initial schooling level

        educ = reshape(zeros(n*(A-16)),n,A-16)
        exper = reshape(zeros(n*(A-16)),n,A-16)

        # Initial level of schooling + assume that initial experience = 0
        for i in 1:n
          educ[i,1] = g16
        end
        d = Dict("e16" => e16, "g16" => g16, "theta"=> theta, "educ" => educ, "exper" => exper)
        return d
    end


  ### I. SOLVING THE MODEL


############## PARAMETERS #############
n=1000
A=26
theta1 = [0.038; 0.033; -0.0005; 0]
theta2 = [0.04; 0.033; -0.0005; -5000]
theta3 = [0.07; 0.055; 0; -5000]
e16_1 = [0 1]
e16_2 = [5000 1]
e16_3 = [5000 .5]

#######################################

        # desired output: optimal rule for school/work decision

        # Algorithm: Backward recursion
        # 0. Initialization: specify rewards, discount factor, last period T (last choice made in T-1),
        # post-terminal value V_T, set t = T-1, define state space
        # V_T can be calculated as a function of 20 possible histories
        # policy function will map from each possible history to an action

        d=simulate(n,A,theta3,e16_3)
        r=1  # wage rate per unit of skill
        delta = 0.95

        function college(educ)
            if educ>12
                return 1
            else
                return 0
            end
        end

        R_1(educ,eps1) = d["e16"][1] .+ d["theta"][4].*college(educ) .+ eps1
        R_2(educ,exper,eps2) = r.*exp(d["e16"][2] .+ d["theta"][1].*educ .+ d["theta"][2].*exper .+ d["theta"][3].*exper.^2 .+ eps2)

        T = A-16 # last period
        z = A-16  # number of state points at which we calculate V()


# generating state space
        state1 = Array{Any}(T,z)    # years of education
        state1[:] = NaN
        state2 = Array{Any}(T,z)    # years of experience
        state2[:] = NaN
        for j in 1:z
            for i in 1:T+1-j
                state1[i,j] = d["g16"]-1 + j
                state2[i,j] = j-1
            end
        end


        # To calculate terminal value, we have to integrate over epsilons: Gaussian quadrature
         nodes, weights = gausslegendre(10)

        V_T = Array{Float64}(2*T)
        choice = Array{Int}(T)
        # Chooses school in the last period
        for j in 1:T
            V_T[j] = dot(R_1(state1[1,j],nodes),weights)*(maximum(nodes)-minimum(nodes))
        end
        # Chooses work in the last period
        for j in T+1:2*T
            V_T[j] = dot(R_2(state1[1,j-T],state2[1,j-T],nodes),weights)*(maximum(nodes)-minimum(nodes))
        end

        if maximum(V_T)[1] <= T
                choice[T] = 1        # choice = 1 => school
               # k = find(V_T .== maximum(V_T)[1])
        else
                choice[T] = 0
               # k = find(V_T .== maximum(V_T)[1]/2)
        end



#1. Recursion Step: given V_T, calculate V_T-1 and optimal choice

    # multiple
        nodes2 = Any[]
        push!(nodes2,repeat(nodes,inner=[1],outer=[10]))  # dim1
        push!(nodes2,repeat(nodes,inner=[10],outer=[1]))
        weights2 = kron(weights,weights)


        V1 = Array{Float64}(T,T) # Array to contain value functions: school
        V1[:] = NaN
        V1[:,T] = V_T[1:T]

        V2 = Array{Float64}(T,T) # Array to contain value functions: work
        V2[:] = NaN
        V2[:,T] = V_T[T+1:2*T]

        # shocks
        e1 = randn(T)
        e2 = randn(T)

        E_V(eps1,eps2,educ,exper,choice) = choice * R_1(educ,eps1) + (1-choice) * R_2(educ,exper,eps2)
        EV_new(eps1,eps2) = E_V(eps1,eps2,state1[1,10],state2[1,10],choice[T])

        for t in T-1:-1:1
            for j in 1:t
                # define alternative-specific functions
                V_1(eps1,eps2) = R_1(state1[11-t,j],e1[j]) .+ delta .* EV_new(eps1,eps2)
                V_2(eps1,eps2) = R_2(state1[11-t,j],state2[11-t,j],e2[j]) .+ delta .* EV_new(eps1,eps2)

                V1[j,t] = dot(V_1(nodes2[1],nodes2[2]),weights2)*(maximum(nodes)-minimum(nodes))
                V2[j,t] = dot(V_2(nodes2[1],nodes2[2]),weights2)*(maximum(nodes)-minimum(nodes))
            end
             if max(maximum(V1[:,t])[1],maximum(V2[:,t])[1]) == maximum(V1[:,t])[1]
                choice[t] = 1        # choice = 1 => school
                k = find(V1[:,t] .== max(maximum(V1[:,t])[1],maximum(V2[:,t])[1]))
            else
                choice[t] = 0
                k = find(V2[:,t] .== max(maximum(V1[:,t])[1],maximum(V2[:,t])[1]))
            end

            educ = state1[t,k]
            exper = state2[t,k]
            choi = choice[t]
            EV_new(eps1,eps2) = E_V(eps1,eps2,state1[t,k],state2[t,k],choi)
        end

    df = DataFrame(Age=16:25, Choice=choice)
    return df
