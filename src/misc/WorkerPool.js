
let workers = [];

export class WorkerPool{
	constructor(){
		
	}

	static getWorker(url, params){
		if (!workers[url]){
			workers[url] = [];
		}

		if (workers[url].length === 0){
			let worker = new Worker(url, params);
			workers[url].push(worker);
		}

		let worker = workers[url].pop();

		return worker;
	}

	static returnWorker(url, worker){
		workers[url].push(worker);
	}
	
};