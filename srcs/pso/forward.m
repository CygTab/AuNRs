function y=forward(in)
global W1 W2 W3 W4 B1 B2 B3 B4
    y1=lrelu(in*W1+B1);
    y2=lrelu(y1*W2+B2);
    y3=lrelu(y2*W3+B3);
    y=y3*W4+B4;
end

function y = lrelu(x)
    alpha = 0.1;
    y = max(alpha.*x, x);
end