

class LRUItem{

	constructor(node){
		this.previous = null;
		this.next = null;
		this.node = node;
		this.timestamp = 0;
	}

}

/**
 *
 * @class A doubly-linked-list of the least recently used elements.
 */
class LRU{

	constructor(){
		// the least recently used item
		this.oldest = null;
		// the most recently used item
		this.newest = null;
		// a list of all items in the lru list
		this.items = new Map();
		this.elements = 0;
	}

	size(){
		return this.elements;
	}

	contains(node){
		return this.items[node.id] == null;
	}

	touch(node, timestamp){

		let item;
		if (!this.items.has(node)) {
			// add to list
			item = new LRUItem(node);
			item.previous = this.newest;
			item.timestamp = timestamp;

			this.newest = item;
			if (item.previous !== null) {
				item.previous.next = item;
			}

			this.items.set(node, item);
			this.elements++;

			if (this.oldest === null) {
				this.oldest = item;
			}
		} else {
			// update in list
			item = this.items.get(node);
			item.timestamp = timestamp;

			if (item.previous === null) {
				// handle touch on oldest element
				if (item.next !== null) {
					this.oldest = item.next;
					this.oldest.previous = null;
					item.previous = this.newest;
					item.next = null;
					this.newest = item;
					item.previous.next = item;
				}
			} else if (item.next === null) {
				// handle touch on newest element
			} else {
				// handle touch on any other element
				item.previous.next = item.next;
				item.next.previous = item.previous;
				item.previous = this.newest;
				item.next = null;
				this.newest = item;
				item.previous.next = item;
			}
		}
	}

	remove(node){
		let lruItem = this.items.get(node);
		
		if (lruItem) {
			if (this.elements === 1) {
				this.oldest = null;
				this.newest = null;
			} else {
				if (!lruItem.previous) {
					this.oldest = lruItem.next;
					this.oldest.previous = null;
				}
				if (!lruItem.next) {
					this.newest = lruItem.previous;
					this.newest.next = null;
				}
				if (lruItem.previous && lruItem.next) {
					lruItem.previous.next = lruItem.next;
					lruItem.next.previous = lruItem.previous;
				}
			}

			// delete this.items[node.id];
			this.items.delete(node);
			this.elements--;
			this.numPoints -= node.numPoints;
		}
	}

	getLRUItem(){
		if (this.oldest === null) {
			return null;
		}
		let lru = this.oldest;

		return lru.node;
	}

	toString(){
		let string = '{ ';
		let curr = this.oldest;
		while (curr !== null) {
			string += curr.node.id;
			if (curr.next !== null) {
				string += ', ';
			}
			curr = curr.next;
		}
		string += '}';
		string += '(' + this.size() + ')';
		return string;
	}
}

export {LRU, LRUItem};