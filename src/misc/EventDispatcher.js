

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

	add(name, callback){
		this.addEventListener(name, callback);
	}

	removeEventListener(name, callback){
		throw "not implemented";
	}

	removeAll(){
		this.listeners = new Map();
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