// Show a choice prompt and return true or false
var message = source.data['message'][0];
if (confirm(message)){
    var choice = 1;
}
else{
    var choice = 0;
}
source.data = {'message' : [message], 'choice': [choice]};
source.change.emit();
