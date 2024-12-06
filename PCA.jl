using LinearAlgebra
using Statistics
function class(l)
    s=0.0
    p=1
    for i in 1:length(l)
        if l[i]>s
            p=i
            s=l[i]
        end
    end
    return p
end
X=[0.975 2.034;2.153 4.127;3.017 5.976;3.885 8.040;5.162 9.987;6.025 12.121]
function mean_center(X)
    s=size(X)
    Xm=zeros(s)
    m=mean(X,dims=1)
    for i in 1:s[1]
        Xm[i,:]=m
    end
    Xcm=X-Xm
    return Xcm
end

function pcas(X)
    Xs=std(X,dims=1)
    t0=[X[i,class(Xs)] for i in 1:size(X)[1]]
    b=true
    t1=[]
    l1=[]
    cont=0
    while b
        lambda=t0'*t0
        l1_=inv(lambda)*t0'*X
        l1=l1_'/norm(l1_)
        t1=X*l1
        delta=lambda-t1'*t1
        if norm(delta)<1e-8||cont>1000000
            b=false
        else
            t0=t1
            b=true
        end
        cont+=1
    end
    X=X-t1*l1'
    return X,t1,l1
end

function PCA(X)
    s=size(X)
    T=zeros(s)
    L=zeros(s[2],s[1])
    d=X'*X
    k=0.0
    for i in 1:s[2]
        k+=d[i,i]
    end
    c=0
    while c<s[2]
        c+=1
        X,t,l=pcas(X)
        T[:,c]=t
        L[:,c]=l
        if (t'*t)/k<0.01
            c=Inf
        end
    end
    V=zeros(size(T)[2])
    v=T'*T
    for i in 1:size(T)[2]
        V[i]=v[i,i]/k
    end
    return T,L,V
end