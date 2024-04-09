from jinja2 import Template

# 定义Jinja2模板字符串
template_string = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="description" content="DIY可视化提供技术支持" />
    <script src="https://cdn.tailwindcss.com"></script>
     <style type="text/css">
        .title { 
          display: block;
          font-size: 2.5em;
          margin-top: 0em;
          margin-bottom: 0em;
          margin-left: 0;
          margin-right: 0;
          font-weight: bold;
        }
        .outerbox{            
            width: 1300px;
            height: 2000px;
            opacity: 1;
            border-radius: 20px;          
            border: 6px solid rgba(238,228,219, 1);
            background: rgba(255, 255, 255, 1);
            box-shadow: 0px 2px 4px  rgba(0, 0, 0, 0.25);
        } 
        .tablebox{
            position: relative;
                left: 10px;
                top: 25px;

             width: 1220px;
             table-layout: fixed;
             {#height: 600px;#}
             border:1px solid rgba(0, 0, 0, 0.73) ;
             cellspacing:"0", 
        }
        th, td {
            border: 1px solid rgba(221,221,221,0.73); /* 设置表格线为1像素宽的黑色实线 */
            text-overflow: ellipsis;
            overflow: hidden;
             vertical-align: top;
            {#background:black#}
            }
        .title2 { 
          {#display: left;#}
          display: block;
          font-size: 1.5em;
          {#margin-top: 0em;
          margin-bottom: 0em;
          margin-left: 0;
          margin-right: 0;
          font-weight: bold;#}
          {#background:black#}
        }

        .similarity_all{
                position: relative;
                left: 15px;
                top: 45px;
                width: 1230px;
                height: 2000px;
                opacity: 1;
                {#background-color: skyblue;#}
                 }


        .sim_overall{
                position: relative;
                left: 31px;
                top: 45px;
                width: 1224px;
                height: 200px;
                opacity: 1;
                background-color: skyblue;
                 }
        .threshold{
                position: relative;
              float: left; 
              left:90px;
              top: 16px;
              margin-right: 310px; 
              opacity: 1;
              font-size: 14px;
                font-weight: 400;
                line-height: 20.27px;
                color: rgba(0, 0, 0, 1);
                text-align: left;
                vertical-align: top;
       }
        .sim_text{
                position: relative;
                left: 0.5px;
                top: 0px;
                width: 1000px;
                height: 32px;
                opacity: 1;
                font-size: 20px;
                font-weight: 700;
                letter-spacing: 0px;
                line-height: 28.96px;
                color: rgba(0, 0, 0, 1);
                text-align: left;
                vertical-align: top;}
        .d2andd3box{
                left: 4px;
                top: 0px;
                width: 1220px;
                height: 300px;
                {#border:10px solid black;#}
                opacity: 1;
                {#background-color: yellow;#}
            }
        .d2d3_names{
                {#position: relative;#}
                {#left: 9.5px;#}
                {#top: 0px;#}
                width: 200px;
                height: 30px;
                opacity: 1;
                font-weight:bold;
                font-size: 18px;
                {#background-color: black;#}
                }
        .div-inline {
              position: relative;
              float: left; 
              left:90px;
              top: 16px;
              margin-right: 180px; 
              opacity: 1;
              font-size: 14px;
                font-weight: 400;
                letter-spacing: 0px;
                line-height: 20.27px;
                color: rgba(0, 0, 0, 1);
                text-align: left;
                vertical-align: top;
            }
        .d2_imgs{
                position: relative;
                float: left; 
                {# margin-right: 160px; #}
                left:0px;
                top: 20px;
                width: 240px;
                height: 240px;
                opacity: 1;}
        .textbox{
                left: 4px;
                top: 49px;
                width: 1220px;
                height: 2000px;
                opacity: 1;
                {#background-color:yellow;#}
            }
        .texts{
                position: relative;
                float: left; 
                {# margin-right: 160px; #}
                left:20px;
                top: 20px;
                width: 240px;
                height: 600px;
                opacity: 1;
                {#background-color:black#}
                }

        .biobox{
                left: 4px;
                top: 49px;
                width: 1220px;
                height: 800px;
                opacity: 1;
                {#background-color: rgba(1, 1, 0, 1);#}
            }

        .commo-textover {
            display:-webkit-box;
            -webkit-box-orient:vertical;
            -webkit-line-clamp:10;
            text-overflow: ellipsis;
            overflow: hidden;
            whitespace: normal;
            word-wrap: break-word;
            {#white-space: nowrap;#}
            }
        .syn_botton {
                {#padding-right: 10px;
                position: absolute;
                right: 0;
                bottom: 5px;#}
                line-height: 1.42857;
                background-color: white;
                cursor: pointer;
                font-weight: bold;
                color: #878ecd;
            }

    </style>


  </head>
  <body>
  {#w-[100.9375rem] h-[234rem] #}
  <div class="relative overflow-hidden gap-10 flex flex-col p-[3.75rem] justify-start items-center bg-[#f3f4f6]" > {#flex flex-col#}
      <h1 class="title" >The molecular analysis results provided by M2UMol</h1> {# this is the title #} 
      <div class="outerbox">
        {# this is the smiles #}
        <div class="relative left-[0.875rem] top-[1.1875rem] w-[77rem] h-[4.125rem] overflow-hidden flex flex-col py-[0.1875rem] justify-start items-start " >
           <h2 class="title2 "> The SMILES of current molecule: </h2>
           <div > {{SMILES}}</div>
        </div>

        {# this is the attention #}
        <div class="relative left-[0.875rem] top-[2.1875rem] w-[77rem] h-[22.125rem] overflow-hidden flex flex-col py-[0.1875rem] justify-start items-start " >
          <h2 class="title2"> The attentions of the molecule: </h2>
          <table class="tablebox"  >
          <tbody>
              <tr>
                    <td ><img src={{attention_img1}} width="90%"> </td>
                    <td ><img src={{attention_img2}} width="90%"></td>
                    <td ><img src={{attention_img3}} width="90%"></td>

              </tr>
              <tr>
                    <td style="text-align:center">Threshold:  {{thre1}} </td>
                    <td style="text-align:center">Threshold:  {{thre2}} </td>
                    <td style="text-align:center">Threshold:  {{thre3}} </td>        
              </tr>

          </tbody>
          </table>
        </div>


        {# this is the similarity #}
        <div class="similarity_all">
            <h2 class="title2"> Top 5 drugs in 4 modalities (2D, 3D, Bio, Text): </h2>
            <table class="tablebox"  >
              <tbody>
                  <tr>
                        <th colspan="5" >2D: </td>                            
                  </tr>
                  <tr>
                        <td >The top 5 drugs are:  </td> 
                        <td colspan="4"> {{d2_drug1}}, {{d2_drug2}}, {{d2_drug3}}, {{d2_drug4}}, {{d2_drug5}} </td>         
                  </tr>
                  <tr>
                        <td style="text-align:center">{{d2_name11}}</td>
                        <td style="text-align:center">{{d2_name22}}</td>
                        <td style="text-align:center">{{d2_name33}}</td>
                        <td style="text-align:center">{{d2_name44}}</td>
                        <td style="text-align:center">{{d2_name55}}</td>

                  </tr>
                  <tr>
                        <td ><img src={{d2_imgs11}} width="100%"></td>
                        <td > <img src={{d2_imgs22}} width="100%">  </td>
                        <td ><img src={{d2_imgs33}} width="100%">  </td>   
                        <td ><img src={{d2_imgs44}} width="100%">   </td>  
                        <td ><img src={{d2_imgs55}} width="100%">    </td>       
                  </tr>

              </tbody>
            </table>
            <table class="tablebox"  >
              <tbody>
                  <tr>
                        <th colspan="5" >3D: </td>

                  </tr>
                  <tr>
                        <td >The top 5 drugs are:  </td> 
                        <td colspan="4"> {{d3_drug1}}, {{d3_drug2}}, {{d3_drug3}}, {{d3_drug4}}, {{d3_drug5}} </td>         
                  </tr>
                  <tr>
                        <td style="text-align:center">{{d3_name11}}</td>
                        <td style="text-align:center">{{d3_name22}}</td>
                        <td style="text-align:center">{{d3_name33}}</td>
                        <td style="text-align:center">{{d3_name44}}</td>
                        <td style="text-align:center">{{d3_name55}}</td>

                  </tr>
                  <tr>
                        <td ><img src={{d3_imgs11}} width="100%"></td>
                        <td > <img src={{d3_imgs22}} width="100%">  </td>
                        <td ><img src={{d3_imgs33}} width="100%">  </td>   
                        <td ><img src={{d3_imgs44}} width="100%">   </td>  
                        <td ><img src={{d3_imgs55}} width="100%">    </td>       
                  </tr>

              </tbody>
            </table>

            <table class="tablebox"  >
              <tbody>
                  <tr>
                        <th colspan="5" >Text: </td>

                  </tr>
                  <tr>
                        <td >The top 5 drugs are:  </td> 
                        <td colspan="4"> {{text_drug1}}, {{text_drug2}}, {{text_drug3}}, {{text_drug4}}, {{text_drug5}} </td>         
                  </tr>
                  <tr>
                        <td style="text-align:center">{{text_name11}}</td>
                        <td style="text-align:center">{{text_name22}}</td>
                        <td style="text-align:center">{{text_name33}}</td>
                        <td style="text-align:center">{{text_name44}}</td>
                        <td style="text-align:center">{{text_name55}}</td>

                  </tr>
                  <tr>
                        <td ><div class="commo-textover" >{{text1}}</div> <div class="syn_botton" onclick="hideContent(this)" data-row="0">Click to Show/Hide</div></td>
                        <td ><div class="commo-textover" >{{text2}}</div> <div class="syn_botton" onclick="hideContent(this)" data-row="0">Click to Show/Hide</div></td>
                        <td ><div class="commo-textover" >{{text3}} </div></td>   
                        <td ><div class="commo-textover" >{{text4}}</div>  <div class="syn_botton" onclick="hideContent(this)" data-row="0">Click to Show/Hide</div></td>  
                        <td ><div class="commo-textover" >{{text5}} <div></td>       
                  </tr>

              </tbody>
            </table>

            <table class="tablebox"  >
              <tbody>
                  <tr>
                        <th colspan="5" >Bio: </td>

                  </tr>
                  <tr>
                        <td >The top 5 drugs are:  </td> 
                        <td colspan="4"> {{bio_drug1}}, {{bio_drug2}}, {{bio_drug3}}, {{bio_drug4}}, {{bio_drug5}}  </td>         
                  </tr>
                  <tr>
                        <td style="text-align:center">{{bio_name11}}</td>
                        <td style="text-align:center">{{bio_name22}}</td>
                        <td style="text-align:center">{{bio_name33}}</td>
                        <td style="text-align:center">{{bio_name44}}</td>
                        <td style="text-align:center">{{bio_name55}}</td>

                  </tr>
                  <tr>
                        <td ><div class="commo-textover" >{{bio1}}</div></td>
                        <td ><div class="commo-textover" >{{bio2}}</td>
                        <td ><div class="commo-textover" >{{bio3}}</div> <div class="syn_botton" onclick="hideContent(this)" data-row="0" >Click to Show/Hide</div></td>   
                        <td ><div class="commo-textover" >{{bio4}}</div> </td>  
                        <td ><div class="commo-textover" >{{bio5}}</div> <div class="syn_botton" onclick="hideContent(this)" data-row="0" >Click to Show/Hide</div></td>       
                  </tr>

              </tbody>
            </table>
            <table class="tablebox"  >
              <tbody>
                  <tr>
                        <th colspan="5" >Top-5 most similar drugs in a comprehensive perspective: </td>                            
                  </tr>
                  <tr>
                        <td >The top 5 drugs are:  </td> 
                        <td colspan="4"> {{DB1}}, {{DB2}}, {{DB3}}, {{DB4}}, {{DB5}}  </td>         
                  </tr>


              </tbody>
            </table>

        </div>



      </div>
    </div>





  </body>

  <script type="text/javascript">


    function hideContent(e) {

        var node = document.getElementById("commo-textover");

        var row = e.getAttribute("data-row");       
        if (!row || row == 0) {
            {#node.style.webkitLineClamp = "";
            node.classList.remove("anpai-hide");
            node.removeAttribute("title");
            node.style.removeProperty('-webkit-line-clamp');#}
            element.className = ''
            return;
        }
    }



</script>
</html>

"""

# 创建Jinja2模板对象

def htmlpainter(smiles,threshold,d2list,d3list,textlist,textdlist,biolist,biotextlist,alllist):
    template = Template(template_string)

    # 使用模板渲染
    html_content = template.render(SMILES=str(smiles),
                                   attention_img1=str(smiles)+"/attention1.jpg",
                                   attention_img2=str(smiles)+"/attention2.jpg",
                                   attention_img3=str(smiles)+"/attention3.jpg",
                                   thre1=threshold[0],
                                   thre2=threshold[1],
                                   thre3=threshold[2],
                                   d2_name11=str(d2list[0]), d2_name22=str(d2list[1]), d2_name33=str(d2list[2]), d2_name44=str(d2list[3]),
                                   d2_name55=str(d2list[4]),
                                   d2_drug1=str(d2list[5]), d2_drug2=str(d2list[6]), d2_drug3=str(d2list[7]), d2_drug4=str(d2list[8]),
                                   d2_drug5=str(d2list[9]),
                                   d2_imgs11=str(smiles)+"/2D-top-0.png", d2_imgs22=str(smiles)+"/2D-top-1.png",
                                   d2_imgs33=str(smiles)+"/2D-top-2.png", d2_imgs44=str(smiles)+"/2D-top-3.png",
                                   d2_imgs55=str(smiles)+"/2D-top-4.png",
                                   d3_name11=str(d3list[5]), d3_name22=str(d3list[6]), d3_name33=str(d3list[7]), d3_name44=str(d3list[8]),
                                   d3_name55=str(d3list[9]),
                                   d3_drug1=str(d3list[0]), d3_drug2=str(d3list[1]), d3_drug3=str(d3list[2]), d3_drug4=str(d3list[3]),
                                   d3_drug5=str(d3list[4]),
                                   d3_imgs11=str(smiles)+"/3D-top-0.png",
                                   d3_imgs22=str(smiles)+"/3D-top-1.png",
                                   d3_imgs33=str(smiles)+"/3D-top-2.png",
                                   d3_imgs44=str(smiles)+"/3D-top-3.png",
                                   d3_imgs55=str(smiles)+"/3D-top-4.png",
                                   text1=str(textlist[0]),
                                   text2=str(textlist[1]),
                                   text3=str(textlist[2]),
                                   text4=str(textlist[3]),
                                   text5=str(textlist[4]),
                                   text_name11=str(textdlist[5]), text_name22=str(textdlist[6]), text_name33=str(textdlist[7]),
                                   text_name44=str(textdlist[8]), text_name55=str(textdlist[9]),
                                   text_drug1=str(textdlist[0]), text_drug2=str(textdlist[1]), text_drug3=str(textdlist[2]), text_drug4=str(textdlist[3]),
                                   text_drug5=str(textdlist[4]),
                                   bio1=str(biotextlist[0]),
                                   bio2=str(biotextlist[1]),
                                   bio3=str(biotextlist[2]),
                                   bio4=str(biotextlist[3]),
                                   bio5=str(biotextlist[4]),
                                   bio_name11=str(biolist[5]), bio_name22=str(biolist[6]), bio_name33=str(biolist[7]), bio_name44=str(biolist[8]),
                                   bio_name55=str(biolist[9]),
                                   bio_drug1=str(biolist[0]), bio_drug2=str(biolist[1]), bio_drug3=str(biolist[2]), bio_drug4=str(biolist[3]),
                                   bio_drug5=str(biolist[4]),

                                   DB1=str(alllist[0]), DB2=str(alllist[1]), DB3=str(alllist[2]), DB4=str(alllist[3]), DB5=str(alllist[4]),
                                   )

    # 打印或者写入文件
    print(html_content)
    # 或者将html_content写入文件
    with open('../M2UMOL/HTMLpainter/'+str(smiles)+'-molecular_information.html', 'w', encoding='utf-8') as file:
        file.write(html_content)