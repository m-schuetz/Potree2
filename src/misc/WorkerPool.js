

class WorkerList{

	constructor(){
		this.count = 0;
		this.list = [];
	}

}

let workers = new Map();

export class WorkerPool{
	constructor(){
		
	}

	static getWorker(url, params){
		if (!workers.has(url)){
			workers.set(url, new WorkerList());
		}

		if (workers.get(url).list.length === 0){
			let worker = new Worker(url, params);
			workers.get(url).list.push(worker);
			workers.get(url).count++;

			console.log(`creating worker. url: ${url}, params: ${Object.entries(params).join()}`);
		}

		let worker = workers.get(url).list.pop();

		return worker;
	}

	static getWorkerCount(url){
		if (!workers.has(url)){
			return 0;
		}else{
			return workers.get(url).count;
		}
	}

	static getAvailableWorkerCount(url){
		if (!workers.has(url)){
			return Infinity;
		}else{
			return workers.get(url).list.length;
		}
	}

	static returnWorker(url, worker){
		workers.get(url).list.push(worker);
	}
	
};