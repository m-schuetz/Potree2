
export class EventDispatcher{

	constructor(){
		this.listeners = {};
	}

	addListener(eventName, callback){
		let listeners = this.listeners;

		if(!listeners[eventName]){
			listeners[eventName] = [];
		}

		listeners[eventName].push(callback);
	}

	dispatch(eventName, event){

		let listeners = this.listeners[eventName];

		if(listeners){
			for(let listener of listeners){
				listener(event);
			}
		}

	}

}