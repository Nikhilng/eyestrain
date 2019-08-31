var express = require('express'); 
var app = express(); 

app.listen(3000, function() { 
    console.log('server running on port 3000'); 
} ) 
  
app.get('/eye', runPy); 
  
let {PythonShell} = require('python-shell')

function runPy(){
    return new Promise(async function(resolve, reject){
          let options = {
          mode: 'text',
          pythonOptions: ['-u'],
          scriptPath: '',//Patto your script
         };

          await PythonShell.run('./eye2.py', options, function (err, results) {
          //On 'results' we get list of strings of all print done in your py scripts sequentially. 
          if (err) throw err;
          console.log('results: ');
          for(let i of results){
                console.log(i, "---->", typeof i)
          }
      resolve(results[1])//I returned only JSON(Stringified) out of all string I got from py script
     });
   })
 } 

function runMain(){
    return new Promise(async function(resolve, reject){
        let r =  await runPy()
        console.log(JSON.parse(JSON.stringify(r.toString())), "Done...!@")//Approach to parse string to JSON.
    })
 }

runMain() //run main function