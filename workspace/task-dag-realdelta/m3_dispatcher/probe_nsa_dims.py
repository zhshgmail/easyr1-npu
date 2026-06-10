import torch, torch_npu
dev="npu:0"
for Dqk,Dv in [(192,128),(512,512),(448,512)]:
    try:
        Tq,Nq,Nkv,sbs,sbc,Skv=64,4,1,64,16,1024
        q=torch.randn(Tq,Nq,Dqk,dtype=torch.bfloat16,device=dev)
        k=torch.randn(Skv,Nkv,Dqk,dtype=torch.bfloat16,device=dev)
        v=torch.randn(Skv,Nkv,Dv,dtype=torch.bfloat16,device=dev)
        tk=torch.arange(sbc,dtype=torch.int32,device=dev).view(1,1,sbc).expand(Tq,Nkv,sbc).contiguous()
        o=torch_npu.npu_nsa_select_attention(q,k,v,tk,Dqk**-0.5,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Skv])
        torch.npu.synchronize()
        print("nsa_select Dqk=%d Dv=%d: RUN ok"%(Dqk,Dv))
    except Exception as e:
        m=str(e).replace("\n"," ")
        code="161002" if "161002" in m else ""
        print("nsa_select Dqk=%d Dv=%d: REJECT %s %s: %s"%(Dqk,Dv,code,type(e).__name__,m[:90]))
print("DONE")
