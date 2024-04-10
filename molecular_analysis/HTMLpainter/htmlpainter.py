from jinja2 import Template

# 定义Jinja2模板字符串
template_string = """
<!doctype html>
<html>
  <head>

    <meta charset="utf-8" />
    <meta name="description" content="The outputs of M2UMol" />
     <title>The outputs of M2UMol</title>
    <script src="https://cdn.tailwindcss.com"></script>
     <style type="text/css">

        .title { 
          font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
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
            height: auto;
            opacity: 1;
            border-radius: 20px;          
            border: 6px solid rgba(238,228,219, 1);
            background: white;
            box-shadow: 0px 2px 4px  rgba(0, 0, 0, 0.25);
        } 
        .tablebox{
            position: relative;
            left: 10px;
            top: 25px;
            background: rgba(255,255,255, 1);
             width: 1220px;
             table-layout: fixed;
             {#height: 600px;#}
             border:1px solid rgba(0, 0, 0, 0.73) ;
             cellspacing:"0", 
        }
        th, td {
            border: 1px solid rgba(221,221,221,0.73); /* 设置表格线为1像素宽的黑色实线 */
            text-overflow: ellipsis;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            overflow: hidden;
             vertical-align: top;
            {#background:black#}
            }
        .table_head{
            background: #bfbfbf;
            text-align: center;
                color: white;
                font-weight: bold;
                font-size: 1.5em;
        }
        .title2 { 
          {#display: left;#}
          font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
          display: block;
          font-weight: bold;
          font-size: 1.5em;
          color:rgba(38,38,38, 1);
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
                height: auto;
                opacity: 1;
                {#background-color: skyblue;#}
                margin-bottom: 100px;
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
            margin-left:10px;
            margin-right:10px;
            display:-webkit-box;
            -webkit-box-orient:vertical;

            text-overflow: ellipsis;
            overflow: hidden;
            whitespace: normal;
            word-wrap: break-word;
            {#white-space: nowrap;#}
            }
        .hide{
        -webkit-line-clamp:9;
        }
        #syn_botton1 {
                {#padding-right: 10px;#}
                {#position: absolute;#}
                position: relative;
                left:60px;

                bottom: 2px;
                line-height: 1.42857;
                background-color: white;
                cursor: pointer;
                font-weight: bold;
                color: #a6a6a6;
            }
        #syn_botton2 {
                {#padding-right: 10px;#}
                {#position: absolute;#}
                position: relative;
                left:60px;
                bottom: 2px;
                line-height: 1.42857;
                background-color: white;
                cursor: pointer;
                font-weight: bold;
                color: #a6a6a6;
            }
        #syn_botton3 {
                {#padding-right: 10px;#}
                {#position: absolute;#}
                position: relative;
                left:60px;
                bottom: 2px;
                line-height: 1.42857;
                background-color: white;
                cursor: pointer;
                font-weight: bold;
                color: #a6a6a6;
            }
        #syn_botton4 {
                {#padding-right: 10px;#}
                {#position: absolute;#}
                position: relative;
                left:60px;
                bottom: 2px;
                line-height: 1.42857;
                background-color: white;
                cursor: pointer;
                font-weight: bold;
                color: #a6a6a6;
            }
        #syn_botton5 {
                {#padding-right: 10px;#}
                {#position: absolute;#}
                position: relative;
                left:60px;
                bottom: 2px;
                line-height: 1.42857;
                background-color: white;
                cursor: pointer;
                font-weight: bold;
                color: #a6a6a6;
            }
        #syn_botton6 {
                {#padding-right: 10px;#}
                {#position: absolute;#}
                position: relative;
                left:60px;
                bottom: 2px;
                line-height: 1.42857;
                background-color: white;
                cursor: pointer;
                font-weight: bold;
                color: #a6a6a6;
            }
        #syn_botton7 {
                {#padding-right: 10px;#}
                {#position: absolute;#}
                position: relative;
                left:60px;
                bottom: 2px;
                line-height: 1.42857;
                background-color: white;
                cursor: pointer;
                font-weight: bold;
                color: #a6a6a6;
            }
        #syn_botton8 {
                {#padding-right: 10px;#}
                {#position: absolute;#}
                position: relative;
                left:60px;
                bottom: 2px;
                line-height: 1.42857;
                background-color: white;
                cursor: pointer;
                font-weight: bold;
                color: #a6a6a6;
            }
        #syn_botton9 {
                {#padding-right: 10px;#}
                {#position: absolute;#}
                position: relative;
                left:60px;
                bottom: 2px;
                line-height: 1.42857;
                background-color: white;
                cursor: pointer;
                font-weight: bold;
                color: #a6a6a6;
            }
        #syn_botton10 {
                {#padding-right: 10px;#}
                {#position: absolute;#}
                position: relative;
                left:60px;
                bottom: 2px;
                line-height: 1.42857;
                background-color: white;
                cursor: pointer;
                font-weight: bold;
                color: #a6a6a6;
            }

    </style>


  </head>
  <body>
  {#w-[100.9375rem] h-[234rem] #}
  <div class="relative overflow-hidden gap-10 flex flex-col p-[3.75rem] justify-start items-center" style=" background: linear-gradient(180deg, rgba(255, 255, 255, 0.5) 0%, rgba(253,226,181, 0.23) 100%);" > {#flex flex-col ;bg-[#f3f4f6] #}
      <h1 class="title" >The molecular analysis results provided by M2UMol</h1> {# this is the title #} 
      <div class="outerbox">


        {# this is the attention #}
        <div class="relative left-[0.875rem] top-[2.1875rem] w-[77rem] h-[29rem] overflow-hidden flex flex-col py-[0.1875rem] justify-start items-start  " >
          <h2 class="title2"> The information of current molecule: </h2>
          <table class="tablebox"  >
              <tbody>
                  <tr>
                        <th class="table_head" colspan="3"  style="background-color:#fad9c6" >SMILES: </td>                            
                  </tr>
                  <tr>

                        <td colspan="3" align="center"> {{SMILES}} </td>                     
                  </tr>
              </tbody>
          </table>
          <br>
          <table class="tablebox"  >
              <tbody>
                  <tr>
                        <th class="table_head" colspan="3" style="background-color:#fad9c6" >The attentions of the molecule: </td>                            
                  </tr>
                  <tr>
                    <td ><img src={{attention_img1}} width="90%" alt="attention_img1"> </td>
                    <td ><img src={{attention_img2}} width="90%" alt="attention_img2"></td>
                    <td ><img src={{attention_img3}} width="90%" alt="attention_img3"></td>

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
                        <th class="table_head" colspan="5" style="background-color:#f3c7c7 " >2D: </td>                            
                  </tr>
                  <tr>
                        <td style="text-align:center">The top 5 drugs are:  </td> 
                        <td colspan="4" style="text-align:center"> {{d2_drug1}}, {{d2_drug2}}, {{d2_drug3}}, {{d2_drug4}}, {{d2_drug5}} </td>         
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
            <br>
            <table class="tablebox"  >
              <tbody>
                  <tr>
                        <th  class="table_head" colspan="5" style="background-color:#d7e3bf" >3D: </td>

                  </tr>
                  <tr>
                        <td style="text-align:center">The top 5 drugs are:  </td> 
                        <td colspan="4" style="text-align:center"> {{d3_drug1}}, {{d3_drug2}}, {{d3_drug3}}, {{d3_drug4}}, {{d3_drug5}} </td>         
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
            <br>
            <table class="tablebox"  >
              <tbody>
                  <tr>
                        <th  class="table_head" colspan="5" style="background-color:#adbacb">Text: </td>

                  </tr>
                  <tr>
                        <td style="text-align:center">The top 5 drugs are:  </td> 
                        <td colspan="4" style="text-align:center"> {{text_drug1}}, {{text_drug2}}, {{text_drug3}}, {{text_drug4}}, {{text_drug5}} </td>         
                  </tr>
                  <tr>
                        <td style="text-align:center">{{text_name11}}</td>
                        <td style="text-align:center">{{text_name22}}</td>
                        <td style="text-align:center">{{text_name33}}</td>
                        <td style="text-align:center">{{text_name44}}</td>
                        <td style="text-align:center">{{text_name55}}</td>

                  </tr>
                  <tr>
                        <td ><div class="commo-textover hide" id="the_text1">{{text1}}</div> <div  id="syn_botton1" onclick="hideContent(this)" data-row="0">Click to Show/Hide</div></td>
                        <td ><div class="commo-textover hide" id="the_text2">{{text2}}</div> <div  id="syn_botton2" onclick="hideContent(this)" data-row="0">Click to Show/Hide</div> </td>
                        <td ><div class="commo-textover hide" id="the_text3">{{text3}}</div> <div  id="syn_botton3" onclick="hideContent(this)" data-row="0">Click to Show/Hide</div> </td>   
                        <td ><div class="commo-textover hide" id="the_text4">{{text4}}</div> <div  id="syn_botton4" onclick="hideContent(this)" data-row="0">Click to Show/Hide</div> </td>  
                        <td ><div class="commo-textover hide" id="the_text5">{{text5}} <div> <div  id="syn_botton5" onclick="hideContent(this)" data-row="0">Click to Show/Hide</div></td>       
                  </tr>

              </tbody>
            </table>
            <br>
            <table class="tablebox"  >
              <tbody>
                  <tr>
                        <th  class="table_head" colspan="5" style="background-color:#b7dddf">Bio: </td>

                  </tr>
                  <tr>
                        <td style="text-align:center">The top 5 drugs are:  </td> 
                        <td colspan="4" style="text-align:center"> {{bio_drug1}}, {{bio_drug2}}, {{bio_drug3}}, {{bio_drug4}}, {{bio_drug5}}  </td>         
                  </tr>
                  <tr>
                        <td style="text-align:center">{{bio_name11}}</td>
                        <td style="text-align:center">{{bio_name22}}</td>
                        <td style="text-align:center">{{bio_name33}}</td>
                        <td style="text-align:center">{{bio_name44}}</td>
                        <td style="text-align:center">{{bio_name55}}</td>

                  </tr>
                  <tr>
                        <td ><div class="commo-textover hide" id="the_text6">{{bio1}}</div> <div  id="syn_botton6" onclick="hideContent(this)" data-row="0">Click to Show/Hide</div></td>
                        <td ><div class="commo-textover hide" id="the_text7">{{bio2}}</div> <div  id="syn_botton7" onclick="hideContent(this)" data-row="0">Click to Show/Hide</div></td>
                        <td ><div class="commo-textover hide" id="the_text8">{{bio3}}</div> <div  id="syn_botton8" onclick="hideContent(this)" data-row="0">Click to Show/Hide</div></td>   
                        <td ><div class="commo-textover hide" id="the_text9">{{bio4}}</div> <div  id="syn_botton9" onclick="hideContent(this)" data-row="0">Click to Show/Hide</div></td>  
                        <td ><div class="commo-textover hide" id="the_text10">{{bio5}}</div> <div  id="syn_botton10" onclick="hideContent(this)" data-row="0" >Click to Show/Hide</div></td>       
                  </tr>

              </tbody>
            </table>
            <br>
            <table class="tablebox"  >
              <tbody>
                  <tr>
                        <th class="table_head" colspan="5" >Top-5 most similar drugs in a comprehensive perspective: </td>                            
                  </tr>
                  <tr>
                        <td style="text-align:center">The top 5 drugs are:  </td> 
                        <td colspan="4" style="text-align:center"> {{DB1}}, {{DB2}}, {{DB3}}, {{DB4}}, {{DB5}}  </td>         
                  </tr>


              </tbody>
            </table>

        </div>



      </div>
    </div>

  <script type="text/javascript">
    var arr = ["1","2","3","4","5","6","7","8","9","10"];

    {#hide the less than ten raw text buttion#}
    function hide_index_fun(id,con_id){
        var the_id = document.getElementById(id);
        let style = window.getComputedStyle(the_id, null);
        let row = Math.ceil(Number(style.height.replace("px", "")) /Number(style.lineHeight.replace("px", "")));//总行高 / 每行行高

        if (row<9){ 

            var conid = document.getElementById(con_id);
            conid.style.display = "none";

        }

    }
    for (var i = 0; i <10;i++){
        var hidetext = "the_text"+arr[i]
        var hidecon = "syn_botton"+arr[i]
        a= hide_index_fun(hidetext,hidecon)

    }

    function hideContent(e) {

        var id = e.getAttribute("id");
        var the_id = id.substr(10)
        var text_node = document.getElementById("the_text"+the_id);
        console.log("aaa");
        console.log(text_node);
        if (text_node.classList.contains("hide")){
        text_node.classList.remove("hide");
        } else{
        text_node.classList.add("hide");
        }

    }
  </script>



  </body>


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