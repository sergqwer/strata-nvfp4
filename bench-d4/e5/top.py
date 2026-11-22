import struct, sys
b=open(sys.argv[1],"rb").read(); n=len(b)//4; a=struct.unpack("%df"%n,b)
o=sorted(range(n),key=lambda i:-a[i])[:5]
print(n, [(i,round(a[i],4)) for i in o], "eos", round(a[248046],4), "nl", round(a[271],4))
