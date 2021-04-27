
function run_settings(imgs) {
    //********* SET UP THESE VARIABLES - MUST BE CORRECT!!!*********************
    // The only input to this script should be a list of images
    first_image=0;
    // imgs = subset_img_list();
    last_image = imgs.length-1;
//    animation_height=777;
//    animation_width=692;
    //animation_startimg=imgs[imgs.length-1];
    //animation_startimg="http://www-hc/~hadhy/CaseStudies/seasia/20171013_Thailand-Bangkok/gpm/2017/10/10/gpm-precip_12hr_20171010T1200Z.png";
    //**************************************************************************

    //=== THE CODE STARTS HERE - no need to change anything below ===
    //=== global variables ====
    theImages = new Array();
    imgs.forEach(get_images); // Loads theImages
    delay = 1000;  //delay between frames in 1/100 seconds
    delay_step = 100;
    delay_max = 4000;
    delay_min = 10;
    current_image = last_image;     //number of the current image
    timeID = null;
    status = 0;            // 0-stopped, 1-playing
    play_mode = 0;         // 0-continuous, 1-normal,  2-swing
    size_valid = 0;

    ani_state.current_image = current_image;
    ani_state.src = theImages[current_image].src;
//    set_active_on_animate(theImages[current_image].src)
};


function get_images(item, index, arr) {
    //===> creates image tags for each image in the imgs array
    theImages[index] = new Image();
    theImages[index].src = arr[index];
};

//function set_active_on_animate(src){
//    pcs = src.split('/')[1].split('.png')[0].split('_');
//    src_vars = {valid: pcs[0],
//                model: pcs[1],
//                region: pcs[2],
//                timeagg: pcs[3],
//                plottype: imgs[0].split('/')[0],
//                plotname: pcs[4],
//                fclt: pcs[5]
//            };
//
//    //app_state[app_state.selected] = src_vars[app_state.selected]
//
//
//    if (app_state.selected == 'valid'){
//
//    } else {
//        var header = document.getElementById(app_state.selected);
//        var btns = header.getElementsByTagName("button")
//        // Loop through all the buttons in the div
//        var i;
//        for (i = 0; i < btns.length; i++) {
//            btn = btns[i];
//            id = btn.getAttribute('id');
//            if (id == src_vars[app_state.selected]) {
//                btn.setAttribute('class', 'button active')
//            } else {
//                btn.setAttribute('class', 'button')
//            }
//        };
//
//    }; // end of if
//};

function animate_fwd() {
    //===> displays image depending on the play mode in forward direction
   current_image++;
   if(current_image > last_image) {
      if (play_mode == 1){
         current_image = last_image;
         ani_state.current_image = current_image;
         ani_state.src = theImages[current_image].src;
         status=0;
         return;
      };                           //NORMAL

      if (play_mode == 0){
         current_image = first_image; //LOOP
         ani_state.current_image = current_image;
         ani_state.src = theImages[current_image].src;
      };

      if (play_mode == 2){
         current_image = last_image;
         ani_state.current_image = current_image;
         ani_state.src = theImages[current_image].src;
         animate_rev();
         return;
      };

   };

   ani_state.current_image = current_image;
   ani_state.src = theImages[current_image].src;
   document.animation.src = theImages[current_image].src;
   document.control_form.frame_nr.value = current_image;
   clearTimeout(timeID);
   status = 1;
   timeID = setTimeout("animate_fwd()", delay);
//   set_active_on_animate(theImages[current_image].src);
   document.getElementById("demo").innerHTML = 'Direct link: <a href=\"' + theImages[current_image].src + '\">'+theImages[current_image].src+'</a>';

}

function animate_rev(){
    //===> displays image depending on the play mode in reverse direction
   current_image--;
   if(current_image < first_image){
      if (play_mode == 1){
         current_image = first_image;
         ani_state.current_image = current_image;
         ani_state.src = theImages[current_image].src;
         status=0;
         return;
      };                           //NORMAL

      if (play_mode == 0){
         current_image = last_image; //LOOP
         ani_state.current_image = current_image;
         ani_state.src = theImages[current_image].src;
      };

      if (play_mode == 2){
         current_image = first_image;
         ani_state.current_image = current_image;
         ani_state.src = theImages[current_image].src;
         animate_fwd();
         return;
      };
   };

   ani_state.current_image = current_image;
   ani_state.src = theImages[current_image].src;
   document.animation.src = theImages[current_image].src;
   document.control_form.frame_nr.value = current_image;
   clearTimeout(timeID);
   status = 1;
   timeID = setTimeout("animate_rev()", delay);
//   set_active_on_animate(theImages[current_image].src);
   document.getElementById("demo").innerHTML = 'Direct link: <a href=\"' + theImages[current_image].src + '\">'+theImages[current_image].src+'</a>';
}

function change_speed(dv){
    //===> changes playing speed by adding to or substracting from the delay between frames
   delay+=dv;
   if(delay > delay_max) delay = delay_max;
   if(delay < delay_min) delay = delay_min;
}

function stop(){
    //===> stop the movie
   if (status == 1) clearTimeout(timeID);
   status = 0;
}

function fwd(){
    //===> "play forward"
   stop();
   status = 1;
   animate_fwd();
}

function go2image(number){
    //===> jumps to a given image number
   stop();
   if (isNaN(number)) number = parseInt(number);
   if (number > last_image) number = last_image;
   if (number < first_image) number = first_image;
   current_image = number;
   ani_state.current_image = current_image;
   ani_state.src = theImages[current_image].src;
   document.animation.src = ani_state.src;
   document.control_form.frame_nr.value = current_image;
//   set_active_on_animate(theImages[current_image].src);
   document.getElementById("demo").innerHTML = 'Direct link: <a href=\"' + theImages[current_image].src + '\">'+theImages[current_image].src+'</a>';
}


function rev(){
    //===> "play reverse"
   stop();
   status = 1;
   animate_rev();
}

function change_mode($i){
    //===> changes play mode (normal, continuous, swing)
    stop();
//    var selectBox = document.getElementById("play_mode_selection");
//    play_mode = selectBox.options[selectBox.selectedIndex].value;
    play_mode = $i;
    animate_fwd();
}


function launch(){
    //===> sets everything once the whole page and the images are loaded (onLoad handler in <body>)
   stop();
   go2image(current_image)

//   current_image = last_image;

   ani_state.current_image = current_image;
   ani_state.src = theImages[current_image].src;
   document.animation.src = theImages[current_image].src;
//   document.animation.width = animation_width;
//   document.animation.height = animation_height;
   document.control_form.frame_nr.value = current_image;
//   set_active_on_animate(theImages[current_image].src);
   document.getElementById("demo").innerHTML = 'Direct link: <a href=\"' + theImages[current_image].src + '\">'+theImages[current_image].src+'</a>';

   // this is trying to set the text (Value property) on the START and END buttons
   // to S(first_image number), E(last_image number). It's supposed (according to
   // JavaScript Authoring Guide) to be a read only value but for some reason
   // it works on win3.11 (on IRIX it doesn't).
   // document.control_form.start_but.value = " FIRST(" + first_image + ") ";
   // document.control_form.end_but.value = " LAST(" + last_image + ") ";
   // this needs to be done to set the right mode when the page is manualy reloaded
    if (status == 1) animate_fwd();
}

function animation(){
    //===> writes the interface into the code where you want it
    // Not used currently

    document.write(" <P><IMG NAME=\"animation\" SRC=\"",animation_startimg,"\" HEIGHT=",animation_height, " WIDTH=", animation_width, "\" ALT=\"[jsMoviePlayer]\">");
    document.write(" <FORM Method=POST Name=\"control_form\"> ");
    document.write("    <INPUT TYPE=\"button\" Name=\"start_but\" Value=\"  FIRST  \" onClick=\"go2image(first_image)\"> ");
    document.write("    <INPUT TYPE=\"button\" Value=\" -1 \" onClick=\"go2image(--current_image)\"> ");
    document.write("    <INPUT TYPE=\"button\" Value=\"BACKWARD\" onClick=\"rev()\"> ");
    document.write("    <INPUT TYPE=\"button\" Value=\" STOP \" onClick=\"stop()\"> ");
    document.write("    <INPUT TYPE=\"button\" Value=\"FORWARD\" onClick=\"fwd()\"> ");
    document.write("    <INPUT TYPE=\"button\" Value=\" +1 \" onClick=\"go2image(++current_image)\"> ");
    document.write("    <INPUT TYPE=\"button\" Name=\"end_but\" Value=\"   LAST   \" onClick=\"go2image(last_image)\"> ");
    document.write(" <BR> ");
    document.write("    OPTIONS:<SELECT NAME=\"play_mode_selection\" onChange=\"change_mode(this.selectedIndex)\"> ");
    document.write("       <OPTION SELECTED VALUE=0>continuous ");
    document.write("       <OPTION VALUE=1>loop once ");
    document.write("       <OPTION VALUE=2>swing ");
    document.write("    </SELECT> ");
    document.write("    IMAGE #:<INPUT TYPE=\"text\" NAME=\"frame_nr\" VALUE=\"0\" SIZE=\"5\" ");
    document.write("     onFocus=\"this.select()\" onChange=\"go2image(this.value)\"> ");
    document.write("    &nbsp;SPEED:<INPUT TYPE=\"button\" Value=\"  -  \" onClick=\"change_speed(delay_step)\"> ");
    document.write("    <INPUT TYPE=\"submit\" Value=\"  +  \" onClick=\"change_speed(-delay_step)\;return false\"> ");
    document.write(" </FORM> ");
    document.write(" </P> ");
};
