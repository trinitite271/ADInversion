/*
 *zhangchang2317@mails.jlu.edu.cn
 *
 *
 *2022/4/11  Changchun
* staggered grid forward finite difference for python NN_inversion
 for elastic wave equation
  libtorch
*/


#include <torch/extension.h>
#include <iostream>
#include <vector>
using namespace torch::indexing;
namespace F = torch::nn::functional;
std::vector<at::Tensor> staggeredfd(
    torch::Tensor inputs,
    torch::Tensor temp,
    torch::Tensor ca,       
    torch::Tensor cl,       
    torch::Tensor cm,       
    torch::Tensor cm1,        
    torch::Tensor b,      
    torch::Tensor b1,      
    torch::Tensor s) {
        auto nt = inputs[0].item().toInt();
        auto nzbc = inputs[1].item().toInt();
        auto nxbc = inputs[2].item().toInt();
        auto dtx = inputs[3].item().toFloat();
        auto ng = inputs[4].item().toInt();
        auto sz = inputs[5].item().toInt();sz--;
        auto sx = inputs[6].item().toInt();sx--;
        auto gz = inputs[7].item().toInt();gz--;
        auto gx = inputs[8].item().toInt();gx--;
        auto dg = inputs[9].item().toInt();
        auto source_type_num = inputs[10].item().toInt();
        auto fd_order_num = inputs[11].item().toInt();
        auto number_elements = nt*ng;
        auto length_geophone = ng*dg;
        auto nt_interval = inputs[12].item().toInt();
        auto nz = inputs[13].item().toInt();
        auto nx = inputs[14].item().toInt();
        auto format_num = inputs[15].item().toInt();
        auto nbc = (nxbc-nx)/2;
        auto num_nt_record = nt/nt_interval;
        auto wavefield_elements = num_nt_record*nx*nz;


    //   Input variables from python numpy: temp ca cl cm b s
    // libtorch Initialising input variables: uu, ww, xx, xz, zz
        torch::Tensor uu = torch::zeros({nzbc,nxbc},torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
        torch::Tensor ww = torch::zeros({nzbc,nxbc},torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
        torch::Tensor xx = torch::zeros({nzbc,nxbc},torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
        torch::Tensor xz = torch::zeros({nzbc,nxbc},torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
        torch::Tensor zz = torch::zeros({nzbc,nxbc},torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
//      libtorch Initialising input variables: fux, fuz, bwx, bwz
        torch::Tensor fux = torch::zeros({nzbc,nxbc},torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
        torch::Tensor fuz = torch::zeros({nzbc,nxbc},torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
        torch::Tensor bwx = torch::zeros({nzbc,nxbc},torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
        torch::Tensor bwz = torch::zeros({nzbc,nxbc},torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
        torch::Tensor illum_div = torch::zeros({nzbc,nxbc},torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
//      libtorch Initialising output variables: seismo_w, seismo_u       
        torch::Tensor seismo_w = torch::zeros({nt,ng},torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
        torch::Tensor seismo_u = torch::zeros({nt,ng},torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
        
        torch::Tensor wavefield_gradient_fux = torch::zeros({nz,nx*num_nt_record},torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
        torch::Tensor wavefield_gradient_fuz = torch::zeros({nz,nx*num_nt_record},torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
        torch::Tensor wavefield_gradient_bwx = torch::zeros({nz,nx*num_nt_record},torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
        torch::Tensor wavefield_gradient_bwz = torch::zeros({nz,nx*num_nt_record},torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
       
//      libtorch zero_vector for free surface zz
        torch::Tensor zero_vector = torch::zeros({1,nxbc},torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
        torch::Tensor geophone_vector = torch::zeros({1,nxbc},torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
        int k;int i;int pad_top;
        if(fd_order_num==22){
            k = nzbc-2; i = nxbc-2; pad_top = 1;}
        else if(fd_order_num==24){
            k = nzbc-4; i = nxbc-4; pad_top = 2;}
        else if(fd_order_num==26){
            k = nzbc-6; i = nxbc-6; pad_top = 3;}
        else if(fd_order_num==28){
            k = nzbc-8; i = nxbc-8; pad_top = 4;}

        float S41 = 1.1250; float S42 = -0.0416666667;
        float S61 = 1.17187; float S62 = -6.51042E-2; float S63 = 4.68750E-3;
        float S81 = 1.19629; float S82 = -7.97526E-2; float S83 = 9.57031E-3; float S84 = -6.97545E-4;

        for( int it = 0; it < nt; it++ ){
if (fd_order_num == 22){ 
    uu = F::pad(temp.index({Slice(1,1+k), Slice(1,1+i)})*(uu.index({Slice(1,1+k), Slice(1,1+i)})) + b.index({Slice(1,1+k), Slice(1,1+i)})*( 
            xx.index({Slice(1,1+k), Slice(1+1,1+1+i)}) - xx.index({Slice(1,1+k), Slice(1,1+i)}) + xz.index({Slice(1,1+k), Slice(1,1+i)}) - xz.index({Slice(1-1,1-1+k), Slice(1,1+i)})), F::PadFuncOptions({1, 1, 1, 1}));                 
    ww = F::pad(temp.index({Slice(1,1+k), Slice(1,1+i)})*(ww.index({Slice(1,1+k), Slice(1,1+i)})) + b1.index({Slice(1,1+k), Slice(1,1+i)})*( 
            xz.index({Slice(1,1+k), Slice(1,1+i)}) - xz.index({Slice(1,1+k), Slice(1-1,1-1+i)}) + zz.index({Slice(1+1,1+1+k), Slice(1,1+i)}) - zz.index({Slice(1,1+k), Slice(1,1+i)})), F::PadFuncOptions({1, 1, 1, 1}));}        
else if(fd_order_num == 24){ 
    uu = F::pad(temp.index({Slice(2,2+k), Slice(2,2+i)})*(uu.index({Slice(2,2+k), Slice(2,2+i)})) + b.index({Slice(2,2+k), Slice(2,2+i)})*( 
            S41*(xx.index({Slice(2,2+k), Slice(2,2+i)}) - xx.index({Slice(2,2+k), Slice(2-1,2-1+i)})) + S42*(xx.index({Slice(2,2+k), Slice(2+1,2+1+i)}) - xx.index({Slice(2,2+k), Slice(2-2,2-2+i)})) +  
            S41*(xz.index({Slice(2,2+k), Slice(2,2+i)}) - xz.index({Slice(2-1,2-1+k), Slice(2,2+i)})) + S42*(xz.index({Slice(2+1,2+1+k), Slice(2,2+i)}) - xz.index({Slice(2-2,2-2+k), Slice(2,2+i)}))), F::PadFuncOptions({2, 2, 2, 2})); 
    ww = F::pad(temp.index({Slice(2,2+k), Slice(2,2+i)})*(ww.index({Slice(2,2+k), Slice(2,2+i)})) + b1.index({Slice(2,2+k), Slice(2,2+i)})*( 
            S41*(xz.index({Slice(2,2+k), Slice(2+1,2+1+i)}) - xz.index({Slice(2,2+k), Slice(2,2+i)})) + S42*(xz.index({Slice(2,2+k), Slice(2+2,2+2+i)}) - xz.index({Slice(2,2+k), Slice(2-1,2-1+i)})) +  
            S41*(zz.index({Slice(2+1,2+1+k), Slice(2,2+i)}) - zz.index({Slice(2,2+k), Slice(2,2+i)})) + S42*(zz.index({Slice(2+2,2+2+k), Slice(2,2+i)}) - zz.index({Slice(2-1,2-1+k), Slice(2,2+i)}))), F::PadFuncOptions({2, 2, 2, 2}));}       
else if(fd_order_num == 26){ 
    uu = F::pad(temp.index({Slice(3,3+k), Slice(3,3+i)})*(uu.index({Slice(3,3+k), Slice(3,3+i)})) + b.index({Slice(3,3+k), Slice(3,3+i)})*( 
            S61*(xx.index({Slice(3,3+k), Slice(3,3+i)}) - xx.index({Slice(3,3+k), Slice(3-1,3-1+i)})) + S62*(xx.index({Slice(3,3+k), Slice(3+1,3+1+i)}) - xx.index({Slice(3,3+k), Slice(3-2,3-2+i)})) + 
            S63*(xx.index({Slice(3,3+k), Slice(3+2,3+2+i)}) - xx.index({Slice(3,3+k), Slice(3-3,3-3+i)})) + S61*(xz.index({Slice(3,3+k), Slice(3,3+i)}) - xz.index({Slice(3-1,3-1+k), Slice(3,3+i)})) + 
            S62*(xz.index({Slice(3+1,3+1+k), Slice(3,3+i)}) - xz.index({Slice(3-2,3-2+k), Slice(3,3+i)})) + S63*(xz.index({Slice(3+2,3+2+k), Slice(3,3+i)}) - xz.index({Slice(3-3,3-3+k), Slice(3,3+i)}))), F::PadFuncOptions({3, 3, 3, 3})); 
    ww = F::pad(temp.index({Slice(3,3+k), Slice(3,3+i)})*(ww.index({Slice(3,3+k), Slice(3,3+i)})) + b1.index({Slice(3,3+k), Slice(3,3+i)})*( 
            S61*(xz.index({Slice(3,3+k), Slice(3+1,3+1+i)}) - xz.index({Slice(3,3+k), Slice(3,3+i)})) + S62*(xz.index({Slice(3,3+k), Slice(3+2,3+2+i)}) - xz.index({Slice(3,3+k), Slice(3-1,3-1+i)})) + 
            S63*(xz.index({Slice(3,3+k), Slice(3+3,3+3+i)}) - xz.index({Slice(3,3+k), Slice(3-2,3-2+i)})) + S61*(zz.index({Slice(3+1,3+1+k), Slice(3,3+i)}) - zz.index({Slice(3,3+k), Slice(3,3+i)})) + 
            S62*(zz.index({Slice(3+2,3+2+k), Slice(3,3+i)}) - zz.index({Slice(3-1,3-1+k), Slice(3,3+i)})) + S63*(zz.index({Slice(3+3,3+3+k), Slice(3,3+i)}) - zz.index({Slice(3-2,3-2+k), Slice(3,3+i)}))), F::PadFuncOptions({3, 3, 3, 3}));} 
else if(fd_order_num == 28){ 
    uu = F::pad(temp.index({Slice(4,4+k), Slice(4,4+i)})*(uu.index({Slice(4,4+k), Slice(4,4+i)})) + b.index({Slice(4,4+k), Slice(4,4+i)})*( 
            S81*(xx.index({Slice(4,4+k), Slice(4,4+i)}) - xx.index({Slice(4,4+k), Slice(4-1,4-1+i)})) + S82*(xx.index({Slice(4,4+k), Slice(4+1,4+1+i)}) - xx.index({Slice(4,4+k), Slice(4-2,4-2+i)})) + 
            S83*(xx.index({Slice(4,4+k), Slice(4+2,4+2+i)}) - xx.index({Slice(4,4+k), Slice(4-3,4-3+i)})) + S84*(xx.index({Slice(4,4+k), Slice(4+3,4+3+i)}) - xx.index({Slice(4,4+k), Slice(4-4,4-4+i)})) + 
            S81*(xz.index({Slice(4,4+k), Slice(4,4+i)}) - xz.index({Slice(4-1,4-1+k), Slice(4,4+i)})) + S82*(xz.index({Slice(4+1,4+1+k), Slice(4,4+i)}) - xz.index({Slice(4-2,4-2+k), Slice(4,4+i)})) + 
            S83*(xz.index({Slice(4+2,4+2+k), Slice(4,4+i)}) - xz.index({Slice(4-3,4-3+k), Slice(4,4+i)})) + S84*(xz.index({Slice(4+3,4+3+k), Slice(4,4+i)}) - xz.index({Slice(4-4,4-4+k), Slice(4,4+i)}))), F::PadFuncOptions({4, 4, 4, 4}));   
    ww = F::pad(temp.index({Slice(4,4+k), Slice(4,4+i)})*(ww.index({Slice(4,4+k), Slice(4,4+i)})) + b1.index({Slice(4,4+k), Slice(4,4+i)})*( 
            S81*(xz.index({Slice(4,4+k), Slice(4+1,4+1+i)}) - xz.index({Slice(4,4+k), Slice(4,4+i)})) + S82*(xz.index({Slice(4,4+k), Slice(4+2,4+2+i)}) - xz.index({Slice(4,4+k), Slice(4-1,4-1+i)})) + 
            S83*(xz.index({Slice(4,4+k), Slice(4+3,4+3+i)}) - xz.index({Slice(4,4+k), Slice(4-2,4-2+i)})) + S84*(xz.index({Slice(4,4+k), Slice(4+4,4+4+i)}) - xz.index({Slice(4,4+k), Slice(4-3,4-3+i)})) + 
            S81*(zz.index({Slice(4+1,4+1+k), Slice(4,4+i)}) - zz.index({Slice(4,4+k), Slice(4,4+i)})) + S82*(zz.index({Slice(4+2,4+2+k), Slice(4,4+i)}) - zz.index({Slice(4-1,4-1+k), Slice(4,4+i)})) + 
            S83*(zz.index({Slice(4+3,4+3+k), Slice(4,4+i)}) - zz.index({Slice(4-2,4-2+k), Slice(4,4+i)})) + S84*(zz.index({Slice(4+4,4+4+k), Slice(4,4+i)}) - zz.index({Slice(4-3,4-3+k), Slice(4,4+i)}))), F::PadFuncOptions({4, 4, 4, 4}));} 

if(source_type_num == 1){ 
         xx.index_put_({sz,sx},xx.index({sz,sx})+s.index({it})); 
         zz.index_put_({sz,sx},zz.index({sz,sx})+s.index({it}));} 
 else if(source_type_num == 2){ 
         uu.index_put_({sz,sx},uu.index({sz,sx})+s.index({it})); 
         uu.index_put_({sz-1,sx},uu.index({sz-1,sx})-s.index({it})); 
         ww.index_put_({sz,sx},ww.index({sz,sx})+s.index({it})); 
         ww.index_put_({sz,sx+1},ww.index({sz,sx+1})+s.index({it}));} 
 else if(source_type_num == 3){ 
         uu.index_put_({sz,sx},uu.index({sz,sx})+s.index({it,0})); 
         ww.index_put_({sz,sx},ww.index({sz,sx})+s.index({it,0}));} 
 else if(source_type_num == 4){ 
         zz.index_put_({sz,sx},zz.index({sz,sx})+s.index({it}));} 
 else if(source_type_num == 5){ 
        //  ww = (ww.index_put_({sz,sx},s.index({it})));} 
        ww.index_put_({sz,sx}, s.index({it,0}));} 

if(fd_order_num == 22){ 
         fux = F::pad(uu.index({Slice(1,1+k), Slice(1,1+i)}) - uu.index({Slice(1,1+k), Slice(1-1,1-1+i)}), F::PadFuncOptions({1, 1, 1, 1})); 
         fuz = F::pad(uu.index({Slice(1+1,1+1+k), Slice(1,1+i)}) - uu.index({Slice(1,1+k), Slice(1,1+i)}), F::PadFuncOptions({1, 1, 1, 1})); 
         bwx = F::pad(ww.index({Slice(1,1+k), Slice(1+1,1+1+i)}) - ww.index({Slice(1,1+k), Slice(1,1+i)}), F::PadFuncOptions({1, 1, 1, 1})); 
         bwz = F::pad(ww.index({Slice(1,1+k), Slice(1,1+i)}) - ww.index({Slice(1-1,1-1+k), Slice(1,1+i)}), F::PadFuncOptions({1, 1, 1, 1}));} 
 else if(fd_order_num == 24){ 
         fux.index({Slice(2,2+k), Slice(2,2+i)}) = S41*(uu.index({Slice(2,2+k), Slice(2+1,2+1+i)}) - uu.index({Slice(2,2+k), Slice(2,2+i)})) + S42*(uu.index({Slice(2,2+k), Slice(2+2,2+2+i)}) - uu.index({Slice(2,2+k), Slice(2-1,2-1+i)})); 
         fuz.index({Slice(2,2+k), Slice(2,2+i)}) = S41*(uu.index({Slice(2+1,2+1+k), Slice(2,2+i)}) - uu.index({Slice(2,2+k), Slice(2,2+i)})) + S42*(uu.index({Slice(2+2,2+2+k), Slice(2,2+i)}) - uu.index({Slice(2-1,2-1+k), Slice(2,2+i)})); 
         bwx.index({Slice(2,2+k), Slice(2,2+i)}) = S41*(ww.index({Slice(2,2+k), Slice(2,2+i)}) - ww.index({Slice(2,2+k), Slice(2-1,2-1+i)})) + S42*(ww.index({Slice(2,2+k), Slice(2+1,2+1+i)}) - ww.index({Slice(2,2+k), Slice(2-2,2-2+i)})); 
         bwz.index({Slice(2,2+k), Slice(2,2+i)}) = S41*(ww.index({Slice(2,2+k), Slice(2,2+i)}) - ww.index({Slice(2-1,2-1+k), Slice(2,2+i)})) + S42*(ww.index({Slice(2+1,2+1+k), Slice(2,2+i)}) - ww.index({Slice(2-2,2-2+k), Slice(2,2+i)}));} 
 else if(fd_order_num == 26){ 
 fux.index({Slice(3,3+k), Slice(3,3+i)}) = S61*(uu.index({Slice(3,3+k), Slice(3+1,3+1+i)}) - uu.index({Slice(3,3+k), Slice(3,3+i)})) + S62*(uu.index({Slice(3,3+k), Slice(3+2,3+2+i)}) - uu.index({Slice(3,3+k), Slice(3-1,3-1+i)}))+ 
         S63*(uu.index({Slice(3,3+k), Slice(3+3,3+3+i)}) - uu.index({Slice(3,3+k), Slice(3-2,3-2+i)})); 
 fuz.index({Slice(3,3+k), Slice(3,3+i)}) = S61*(uu.index({Slice(3+1,3+1+k), Slice(3,3+i)}) - uu.index({Slice(3,3+k), Slice(3,3+i)})) + S62*(uu.index({Slice(3+2,3+2+k), Slice(3,3+i)}) - uu.index({Slice(3-1,3-1+k), Slice(3,3+i)}))+ 
         S63*(uu.index({Slice(3+3,3+3+k), Slice(3,3+i)}) - uu.index({Slice(3-2,3-2+k), Slice(3,3+i)})); 
 bwx.index({Slice(3,3+k), Slice(3,3+i)}) = S61*(ww.index({Slice(3,3+k), Slice(3,3+i)}) - ww.index({Slice(3,3+k), Slice(3-1,3-1+i)})) + S62*(ww.index({Slice(3,3+k), Slice(3+1,3+1+i)}) - ww.index({Slice(3,3+k), Slice(3-2,3-2+i)}))+ 
         S63*(ww.index({Slice(3,3+k), Slice(3+2,3+2+i)}) - ww.index({Slice(3,3+k), Slice(3-3,3-3+i)})); 
 bwz.index({Slice(3,3+k), Slice(3,3+i)}) = S61*(ww.index({Slice(3,3+k), Slice(3,3+i)}) - ww.index({Slice(3-1,3-1+k), Slice(3,3+i)})) + S62*(ww.index({Slice(3+1,3+1+k), Slice(3,3+i)}) - ww.index({Slice(3-2,3-2+k), Slice(3,3+i)}))+ 
         S63*(ww.index({Slice(3+2,3+2+k), Slice(3,3+i)}) - ww.index({Slice(3-3,3-3+k), Slice(3,3+i)}));} 
 else if(fd_order_num == 28){ 
 fux.index({Slice(4,4+k), Slice(4,4+i)}) = S81*(uu.index({Slice(4,4+k), Slice(4+1,4+1+i)}) - uu.index({Slice(4,4+k), Slice(4,4+i)})) + S82*(uu.index({Slice(4,4+k), Slice(4+2,4+2+i)}) - uu.index({Slice(4,4+k), Slice(4-1,4-1+i)}))+ 
         S83*(uu.index({Slice(4,4+k), Slice(4+3,4+3+i)}) - uu.index({Slice(4,4+k), Slice(4-2,4-2+i)})) + S84*(uu.index({Slice(4,4+k), Slice(4+4,4+4+i)}) - uu.index({Slice(4,4+k), Slice(4-3,4-3+i)})); 
 fuz.index({Slice(4,4+k), Slice(4,4+i)}) = S81*(uu.index({Slice(4+1,4+1+k), Slice(4,4+i)}) - uu.index({Slice(4,4+k), Slice(4,4+i)})) + S82*(uu.index({Slice(4+2,4+2+k), Slice(4,4+i)}) - uu.index({Slice(4-1,4-1+k), Slice(4,4+i)}))+ 
         S83*(uu.index({Slice(4+3,4+3+k), Slice(4,4+i)}) - uu.index({Slice(4-2,4-2+k), Slice(4,4+i)})) + S84*(uu.index({Slice(4+4,4+4+k), Slice(4,4+i)}) - uu.index({Slice(4-3,4-3+k), Slice(4,4+i)})); 
 bwx.index({Slice(4,4+k), Slice(4,4+i)}) = S81*(ww.index({Slice(4,4+k), Slice(4,4+i)}) - ww.index({Slice(4,4+k), Slice(4-1,4-1+i)})) + S82*(ww.index({Slice(4,4+k), Slice(4+1,4+1+i)}) - ww.index({Slice(4,4+k), Slice(4-2,4-2+i)}))+ 
         S83*(ww.index({Slice(4,4+k), Slice(4+2,4+2+i)}) - ww.index({Slice(4,4+k), Slice(4-3,4-3+i)})) + S84*(ww.index({Slice(4,4+k), Slice(4+3,4+3+i)}) - ww.index({Slice(4,4+k), Slice(4-4,4-4+i)})); 
 bwz.index({Slice(4,4+k), Slice(4,4+i)}) = S81*(ww.index({Slice(4,4+k), Slice(4,4+i)}) - ww.index({Slice(4-1,4-1+k), Slice(4,4+i)})) + S82*(ww.index({Slice(4+1,4+1+k), Slice(4,4+i)}) - ww.index({Slice(4-2,4-2+k), Slice(4,4+i)}))+ 
         S83*(ww.index({Slice(4+2,4+2+k), Slice(4,4+i)}) - ww.index({Slice(4-3,4-3+k), Slice(4,4+i)})) + S84*(ww.index({Slice(4+3,4+3+k), Slice(4,4+i)}) - ww.index({Slice(4-4,4-4+k), Slice(4,4+i)}));}
       xx=temp * (xx) + (ca * (fux) + cl * (bwz))*dtx;
       zz=temp * (zz) + (ca * (bwz) + cl * (fux))*dtx;
       xz=temp * (xz) + (cm1 * (fuz + bwx))*dtx;

    //    zz.row(pad_top) = zero_vector;
       zz.index({pad_top,Slice(None)})=0.0;
    //    geophone_vector=ww.row(gz);
       seismo_w.index({it,Slice(None)}) = (ww.index({gz,Slice(gx, gx+length_geophone-1, dg)}));
    //    geophone_vector=uu.row(gz);
       seismo_u.index({it,Slice(None)}) = (uu.index({gz,Slice(gx, gx+length_geophone-1, dg)}));
       if(it%nt_interval==0){
        illum_div = illum_div+torch::pow(fux+bwz,2);
    //    wavefield_gradient_fux.index({Slice(None),Slice(nx*it/nt_interval,nx*it/nt_interval + nx)})=fux.index({Slice(pad_top+1,pad_top+1+nz),Slice(nbc,nbc + nx)});
    //    wavefield_gradient_fuz.index({Slice(None),Slice(nx*it/nt_interval,nx*it/nt_interval + nx)})=fuz.index({Slice(pad_top+1,pad_top+1+nz),Slice(nbc,nbc + nx)});
    //    wavefield_gradient_bwx.index({Slice(None),Slice(nx*it/nt_interval,nx*it/nt_interval + nx)})=bwx.index({Slice(pad_top+1,pad_top+1+nz),Slice(nbc,nbc + nx)});
//        wavefield_gradient_bwz.index({Slice(None),Slice(nx*it/nt_interval,nx*it/nt_interval + nx)})=bwz.index({Slice(pad_top+1,pad_top+1+nz),Slice(nbc,nbc + nx)});  
}
// std::cout<<seismo_w.__dispatch__version()<<std::endl;
}

  return {seismo_u,
          seismo_w,
          illum_div};
}


PYBIND11_MODULE(libtorch_staggerfd_cuda, m) {
  m.def("forward", &staggeredfd, "forward");
}