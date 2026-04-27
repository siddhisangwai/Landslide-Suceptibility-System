
        var x=document.getElementById("ans");
        function fun(){
            if(navigator.geolocation){
                navigator.geolocation.watchPosition(show);
            }
            else{
                x.innerHTML="Your Browser doesn't support location."
            }
        }
        function show(position){
            x.innerHTML = "Latitude: " + position.coords.latitude + 
                   "<br>Longitude: " + position.coords.longitude;

                   var pos = position.coords.latitude + "," + position.coords.longitude;
 }