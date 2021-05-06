

export class EventDispatcher{

	constructor(){
		this.listeners = new Map();
	}

	addEventListener(name, callback){

		let list = this.listeners.get(name);

		if(!list){
			list = [];

			this.listeners.set(name, list);
		}

		list.push(callback);

	}

	removeEventListener(name, callback){
		throw "not implemented";
	}

	dispatch(name, data){
		let list = this.listeners.get(name);

		if(!list){
			return;
		}

		for(let callback of list){
			callback(data);
		}
	}

}