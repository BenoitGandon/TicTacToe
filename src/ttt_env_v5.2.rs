use std::io;
//v2: Edge own by in_node, remove edge hashmap
//		Correct env legal moves
//v3: Add action_values and evaluate_leaf based on value based action selection instead of random
//v4: 	- Add strategy selector on overboard: malus calculated with over_board_value
//		- simulated value weight by game length
//		- battle with baseline
//		- Add simulation_method as agent parameter
//		- over_board_value: OnGoing value replaced with subboard evaluation
//		- random selection weighted on prior
//v5:
//		- Add overboard_evaluation to estimate board value
//		TODO:
//		- if len(moves)> 9, force focus on best moves (top9?)
//		- fine tuning cpuct high?
//		- test evaluation_method on differents setups
//
//v5.2:
//		- Save TttEnv as game_state in Node, add Clone trait
//		- Add MCTS cleaning: 
//			*option1: For each Node, if not root and ref InNode is unknow> delete Node
//			*option2: For each Node, check compatibility of overBoard with Root overBoard > choosen
//		- node_id : change last move for available grid move (0-9 or A for any) to avoid duplicated position if last move give free move
//		- initialize node_map with capacity > 500K to avoid reallocation
//		- action_values: scale move value with overboard evaluation
//		- rename and seperate function for overboard prior/value calculation and subboard prior/value calculation
//		
//		ISSUES:
//		- freeze when Hasmap node > 460K entries TBC [x] > initialize capacity on hashmap + clean MCTS
//		- underevaluation of position that give free move but win a subgrid? increase cpuct? do short simulation before evaluation?
//		- cleaning MCTS is time consuming. partial clean with timer?
//		
//v5.2.1:
//		- remove MCTS cleaning, increase max simulation

use std::fmt;
use std::collections::HashMap;
use rand::seq::SliceRandom;
use rand::Rng;
use std::time; //use for benchmark
use std::cmp;

#[derive(Copy, Clone, PartialEq, Debug)]
enum GameStatus {
	Won(i8), //1: player1 won, -1:player2 won
	Draw,
	OnGoing,
}

impl GameStatus {
    pub const fn is_Won(&self) -> bool {
        matches!(*self, GameStatus::Won(_))
    }
}

#[derive(Debug, Clone)]
struct TttEnv {
	over_board: [GameStatus;9],
	board: [[i8;9];9],
	last_move: Option<(u8,u8)>,
	player_turn: i8,		//1: player1, -1:player2
	pub status: GameStatus,
}

//provide default values for your struct by implementing the Default trait
impl Default for TttEnv {
	fn default() -> TttEnv {
		TttEnv {
			over_board: [GameStatus::OnGoing;9],
			board: [[0i8;9];9],
			last_move: None,
			player_turn: 1,
			status: GameStatus::OnGoing,
		}
	}
}

//Display Trait for TttEnv
impl fmt::Display for TttEnv {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "Game status: {:?}, last_move: {:?}, player_turn: {}\n", self.status, self.last_move, self.player_turn);
		write!(f, "board: \n{:?}", self.board)
	}
}

impl TttEnv {
	// Return vector of allowed move as tuple
	fn legal_moves(&self) ->   Vec<(u8,u8)> {
		if self.status != GameStatus::OnGoing {
			let moves :Vec<(u8,u8)> = Vec::new();//game is over, no legals_moves are allowed
			return moves} 
		match self.last_move {
			None => return (0..81).map(|x| (x/9,x%9)).collect(),
			Some(gn) => {
			    let mut moves :Vec<(u8,u8)> = Vec::new();
				if self.over_board[gn.1 as usize] == GameStatus::OnGoing {
					moves = self.board[gn.1 as usize]
							.iter()
							.enumerate()
							.filter(|(_idx, &val)| val==0)
							.map(|(idx, _x)| (gn.1,idx as u8))
							.collect();
				}
				else {
					for (idx_b, b) in self.board.iter().enumerate() {
						if self.over_board[idx_b] == GameStatus::OnGoing {
							let mut b_moves :Vec<(u8,u8)>;
							b_moves = b.iter()
										.enumerate()
										.filter(|(_idx, &val)| val==0)
										.map(|(i, _x)| (idx_b as u8,i as u8))
										.collect();
							moves.append(& mut b_moves);
						}
					}
				}
				return moves
			},
		}
	}
	fn switch_players(&mut self) -> () {
		self.player_turn *= -1;
	}
	//check status of subboard and return current GameStatus
	fn check_board_status(&self, &sb:&[i8;9]) -> GameStatus {
		let mut status = GameStatus::OnGoing;
		if sb[0]==sb[1] && sb[0]==sb[2] && sb[0]!=0 {status = GameStatus::Won(sb[0])}	//line1 complete
		else if sb[3]==sb[4] && sb[3]==sb[5] && sb[3]!=0 {status = GameStatus::Won(sb[3]);} //line2 complete
		else if sb[6]==sb[7] && sb[6]==sb[8] && sb[6]!=0 {status = GameStatus::Won(sb[6]);} //line3 complete
		else if sb[0]==sb[3] && sb[0]==sb[6] && sb[0]!=0 {status = GameStatus::Won(sb[0]);} //column1 complete
		else if sb[1]==sb[4] && sb[1]==sb[7] && sb[1]!=0 {status = GameStatus::Won(sb[1]);} //column2 complete
		else if sb[2]==sb[5] && sb[2]==sb[8] && sb[2]!=0 {status = GameStatus::Won(sb[2]);} //column3 complete
		else if sb[0]==sb[4] && sb[0]==sb[8] && sb[0]!=0 {status = GameStatus::Won(sb[0]);} //diag1 complete
		else if sb[2]==sb[4] && sb[2]==sb[6] && sb[2]!=0 {status = GameStatus::Won(sb[2]);} //diag2 complete
		else if !sb.contains(&0i8) {status = GameStatus::Draw;} //board full
		
		return status
	}
	//check status of over_board and update game status
	fn check_game_status(&mut self) -> () {
		let mut status = GameStatus::OnGoing;
		if self.over_board[0]==self.over_board[1] && self.over_board[0]==self.over_board[2] && self.over_board[0].is_Won()
			{status = self.over_board[0];}	//line1 complete
		else if self.over_board[3]==self.over_board[4] && self.over_board[3]==self.over_board[5] && self.over_board[3].is_Won()
			{status = self.over_board[3];}	//line2 complete
		else if self.over_board[6]==self.over_board[7] && self.over_board[6]==self.over_board[8] && self.over_board[6].is_Won()
			{status = self.over_board[6];}	//line3 complete
		else if self.over_board[0]==self.over_board[3] && self.over_board[0]==self.over_board[6] && self.over_board[0].is_Won()
			{status = self.over_board[0];}	//column1 complete
		else if self.over_board[1]==self.over_board[4] && self.over_board[1]==self.over_board[7] && self.over_board[1].is_Won()
			{status = self.over_board[1];}	//column2 complete
		else if self.over_board[2]==self.over_board[5] && self.over_board[2]==self.over_board[8] && self.over_board[2].is_Won()
			{status = self.over_board[2];}	//column3 complete
		else if self.over_board[0]==self.over_board[4] && self.over_board[0]==self.over_board[8] && self.over_board[0].is_Won()
			{status = self.over_board[0];}	//diag1 complete
		else if self.over_board[2]==self.over_board[4] && self.over_board[2]==self.over_board[6] && self.over_board[2].is_Won()
			{status = self.over_board[2];}	//diag2 complete
		else if !self.over_board.contains(&GameStatus::OnGoing) {status = GameStatus::Draw;} //board full
		
		self.status = status
	}
	// Make current player play 'gn' move and update board game and status
	fn step(&mut self, gn:(u8,u8)) -> () {
		let moves = self.legal_moves();
		// check if move is allowed
		if ! moves.contains(&gn) {
			panic!("Error move {:?} is not allowed at this point\n {:?}", &gn, &self);
		}
		//update game board
		self.board[gn.0 as usize][gn.1 as usize] = self.player_turn;
		//save last move
		self.last_move = Some(gn);
		
		//update player turn
		self.switch_players();
		
		//update over board
		self.over_board[gn.0 as usize] = self.check_board_status(&self.board[gn.0 as usize]);
		//update game status
		self.check_game_status();
	}
	// Generate unique String id link to the current game position
	fn generate_id(&self) -> String {
		let mut board_id = String::from("");
		for (g, sub_status) in self.over_board.iter().enumerate(){
			match sub_status {
				GameStatus::Won(1) => board_id += "A_",
				GameStatus::Won(-1) => board_id += "B_",
				GameStatus::Draw => board_id += "D_",
				GameStatus::OnGoing => {
					let mut subboard_id = String::from("");
					for n in self.board[g] {
						match n {
							0 => subboard_id += "0",
							-1 => subboard_id += "B",
							1 => subboard_id += "A",
							_ => (),
						}
					}
					board_id = board_id + &subboard_id + "_";
				},
				_ => (),
			}
		}
		//add grid available move to id (same gameState can happen with different allowed moves)
		match self.last_move {
			Some((_g,n)) => {
				if self.over_board[n as usize] == GameStatus::OnGoing { //last move give available move in subgrid n only
					board_id = board_id + &n.to_string();
				}
				else {
					board_id = board_id + "A"; //last move give move in (A)ny subgrid 
				}
			},
			None => board_id += "-"
		}
		//add player turn
		match self.player_turn {
			1 => board_id += "A",
			-1 => board_id += "B",
			_ => (),
		}
		return board_id
	}
	//set the game to specific state
	fn set_state(&mut self, board:[[i8;9];9], last_move:Option<(u8,u8)>, player_turn: i8) -> () {
		self.board = board;
		self.last_move = last_move;
		self.player_turn = player_turn;
		//update overboard
		for (idx, &subboard) in self.board.iter().enumerate(){
			let status = self.check_board_status(&subboard);
			self.over_board[idx] = status;
		}
		//update game status
		self.check_game_status();
	}
	//save a copy of the current game state
	fn save_state(&self) -> ([[i8;9];9], Option<(u8,u8)>, i8, GameStatus) {
		let copy_board = self.board;
		let copy_last_move = self.last_move;
		let copy_player_turn = self.player_turn;
		let copy_status = self.status;
		
		return (copy_board, copy_last_move, copy_player_turn, copy_status)
	}
	
}

/* ########################################### MonteCarloTreeSearch definition ######################################################### */
#[derive(Debug)]
struct Node {
	//save a game state
	game_state: TttEnv,
	id: String,
	edges_out: Vec<Edge>
}
impl Node {
	fn is_leaf(&self) -> bool {
		self.edges_out.len() == 0
	}
	
	fn is_compatible(&self, ref_overboard: [GameStatus;9]) -> bool {
		//check if Node game state in compatible with reference overboard values
		for (b_status, ref_status) in self.game_state.over_board.iter().zip(ref_overboard.iter()) {
			if (*b_status != *ref_status) & (*ref_status != GameStatus::OnGoing) & (*b_status != GameStatus::OnGoing) {
				//eprintln!("incompatible status: {:?}, {:?}", *b_status, *ref_status);
				return false
			}
		}
		return true
	}
}

#[derive(Debug)]
struct Edge {
	out_node_id: String,
	action: (u8,u8),
	
	//stats
	N: u32,		// nb of time node have been reach
	W: f32,		// total value of next state
	Q: f32,		// mean value of the next state = W/N
	P: f32,		// prior value (base on NN or opportunity move)
}

#[derive(Debug)]
struct MCTS {
	root_id : String,
	node_map : HashMap<String, Node>,
	cpuct : f32, //exploration parameter
	
}
impl MCTS {
	//select a Node with no child that maximize Q+U
	// return : Node id reference, path to Node as vector of (Node, EdgeIdx)
	fn move_to_leaf(&self)-> (String, Vec<(String, usize)>) {
		let mut current_node = self.node_map.get(&self.root_id).unwrap();
		let mut path_to_node: Vec<(String, usize)> = Vec::new();
		while !current_node.is_leaf() {
			//eprintln!("{:?}", &current_node);
			let mut max_QU = -99999.0;
			//get sum of N for childrens
			let mut Nb:u32 = 0;
			for edge in current_node.edges_out.iter() {
				Nb += edge.N;
			}
			
			let mut simulation_edge_idx = 0;
			let mut simulation_node_id = &current_node.edges_out[0].out_node_id;
			
			for (edge_idx, edge) in current_node.edges_out.iter().enumerate() {
				let Q = edge.Q;
				let U = self.cpuct * edge.P * (Nb as f32).sqrt() / (1.0 + edge.N as f32);
				if Q + U > max_QU {
					max_QU = Q + U;
					simulation_edge_idx = edge_idx;
					simulation_node_id = &edge.out_node_id;
				}
			}
			//save path to Node
			path_to_node.push((current_node.id.to_string(), simulation_edge_idx));
			//select node connected to best edge
			current_node = self.node_map.get(simulation_node_id).unwrap();
		}
		return (current_node.id.to_string(), path_to_node)
	}
	//back propagate the Node value to parents Nodes
	fn back_fill(&mut self, leaf_id:&str, value:f32, path_to_node:Vec<(String,usize)>) -> () {
		let player_turn = self.node_map.get(leaf_id).unwrap().game_state.player_turn;
		let mut direction: f32;
		for (node_id, edge_idx) in path_to_node {
			let node = self.node_map.get_mut(&node_id).unwrap(); //get mutable ref of node
			if node.game_state.player_turn == player_turn {direction = 1.0} else {direction = -1.0};
			node.edges_out[edge_idx].N += 1;
			node.edges_out[edge_idx].W += value * direction;
			node.edges_out[edge_idx].Q = node.edges_out[edge_idx].W / node.edges_out[edge_idx].N as f32;
		}
	}
	//from specified environment, if Node do not exist, create Node and add it to the graph. return Node reference as String
	fn add_node(&mut self, env:&TttEnv) -> String {
		let node_id = env.generate_id();
		if !self.node_map.contains_key(&node_id) {
			let node = Node {game_state:env.clone(),
					id:node_id.to_string(),
					edges_out: Vec::with_capacity(9), //most of situation
					};
			self.node_map.insert(node_id.to_string(), node);
		}
		return node_id
	}
	//return game state from specified node. Panic if node doesn't exist
	fn get_leaf_game_state(&self, node_id:&str) -> (TttEnv) {
		let game_state = self.node_map.get(node_id).unwrap().game_state.clone();
		return game_state
	}
	//clean MCTS from all nodes incompatibles with root
	fn clean_tree(&mut self) -> () {
		let ref_overboard = self.node_map.get(&self.root_id).unwrap().game_state.over_board;
		self.node_map.retain(|id, node| {
			let compatible = node.is_compatible(ref_overboard);
			//if !compatible {eprintln!("{:?} | {:?}", &node.game_state.over_board, &ref_overboard);}
			compatible
		});
		//eprintln!("Number of Node after cleaning: {}", self.node_map.len())
	}
}

#[derive(Debug)]
struct Agent {
	mcts : MCTS,
	nb_simulations: u32,
	nb_evaluation_sim: u32,
	evaluation_method: fn(&Node, u32) -> f32,
	act_timer_ms: u64,
	clean_mcts: bool,
}

impl Agent {
	//apply policy to retrieve best action and value from a gameState
	fn act(&mut self, env:&TttEnv) -> ((u8,u8), f32, f32) {
		
		//init timer
		let start_time = time::Instant::now();
		
		//find or create current state in graph and set as root
		let root_id = self.mcts.add_node(env);
		//set root
		self.mcts.root_id = root_id;
		
		//clean MCTS tree
		if self.clean_mcts {self.mcts.clean_tree();}
		
		for nb_sim in 1..self.nb_simulations{
			let (leaf_id, path) = self.mcts.move_to_leaf(); //select a leaf from tree
			//eprintln!("simulation nb: {:?}, moved to leaf", nb_sim);
			let value = self.evaluate_leaf(&leaf_id); //evaluate leaf value with random plays for the current leaf player also expand leaf if game is still OnGoing
			//eprintln!("simulation nb: {:?}, evaluated leaf {}", nb_sim, value);
			
			self.mcts.back_fill(&leaf_id, value, path); //backfill tree with evaluated value
			//eprintln!("simulation nb: {:?}, backfilled tree", nb_sim);
			
			//check timer limit
			let elapsed_time = start_time.elapsed();
			//eprintln!("simulation nb: {:?}, start", nb_sim);
			if elapsed_time > time::Duration::from_millis(self.act_timer_ms) {
				eprintln!("stop after {:?} simulations", nb_sim);
				break
				}
		}
		
		let (action, mcts_value, _prior) = self.select_best_action();
		let mut estimated_value = f32::NAN;
		
		if ESTIMATE_VALUE {
			let root_node = self.mcts.node_map.get(&self.mcts.root_id).unwrap();
			estimated_value = (self.evaluation_method)(root_node, self.nb_evaluation_sim)
		}
		
		return (action, mcts_value, estimated_value)
	}
	//evaluate leaf with random actions until game is finish
	fn evaluate_leaf(&mut self, leaf_id:&str) -> f32 {
		//get leaf reference
		let node = self.mcts.node_map.get(leaf_id).unwrap();
		match node.game_state.status {
			GameStatus::Won(player) => {if player==node.game_state.player_turn {return WIN} else {return LOSE}},
			GameStatus::Draw => return DRAW,
			GameStatus::OnGoing => {
				let value = (self.evaluation_method)(node, self.nb_evaluation_sim);
				//expand leaf
				self.expand_leaf(leaf_id); //expand leaf if the game is not over
				return value
			},
		}
	}
	//select best action from the mcts root
	fn select_best_action(&self) -> ((u8,u8), f32, f32) {
		let root = self.mcts.node_map.get(&self.mcts.root_id).unwrap();
		//pick action with max N, value=Q
		let mut best_idx = 0usize;
		let mut max_n = 0u32;
		let mut best_q = -1.0;
		
		for (idx, edge) in root.edges_out.iter().enumerate() {
			if (edge.N > max_n) | ((edge.N==max_n) & (edge.Q > best_q)) {
				best_idx = idx;
				max_n = edge.N;
				best_q = edge.Q;
			}
			//eprintln!("{:?}, N:{}, Q:{:.3}, P:{:.3}", edge.action, edge.N, edge.Q, edge.P);
		}
		//if root.edges_out.len() < 2 {eprintln!("Root: {:?}", &root);}
		let best_edge = &root.edges_out[best_idx];
		//let estimated_value = (self.evaluation_method)(root, self.nb_evaluation_sim); //estimated_value is actually used as prior
		
		return (best_edge.action, best_edge.Q, best_edge.P)
	}
	
	//Create a Node and Edge for each available move and add it to the mcts graph
	fn expand_leaf(&mut self, leaf_id:&str)->() {
		//get leaf data
		let game_state = self.mcts.get_leaf_game_state(leaf_id);
		//create test_env as copy of leaf state
		let mut test_env = game_state.clone();
		let moves = action_values(&test_env);
		let mut edges:Vec<Edge> = Vec::new();
		//prior based on action value
		for (action, _value, prior) in moves {
			//restore test_env
			let mut test_env = game_state.clone();
			
			//simulate action in test_env
			test_env.step(action);
			//add node if not in graph
			let node_out_id = self.mcts.add_node(&test_env);
			
			//create new edge
			let edge = Edge {
				out_node_id: node_out_id,
				action: action,
				
				//stats
				N: 0,		// nb of time node have been reach
				W: 0.0,		// total value of next state
				Q: 0.0,		// mean value of the next state = W/N
				P: prior,	// prior value (base on NN or opportunity move)
				};
			
			//add edge to graph as link between leaf and node
			edges.push(edge);
		}
		//add edges to leaf
		let leaf = self.mcts.node_map.get_mut(leaf_id).unwrap();
		leaf.edges_out.append(&mut edges)
	}
	
	fn reset_mcts(&mut self, env:&TttEnv)->() {
		let root_id = env.generate_id();
		let game_root = Node {game_state:env.clone(),
								id:root_id.to_string(),
								edges_out: Vec::with_capacity(9),
								};
		
		let mcts = MCTS {root_id: root_id.to_string(), //create a copy because root id will be consume in hashmap
							node_map: HashMap::from([(root_id, game_root)]),
							cpuct: self.mcts.cpuct,
							};
		//reset mcts
		self.mcts = mcts;
		}
}

fn new_Agent(env:&TttEnv, cpuct:f32, nb_evaluation_sim:u32, evaluation_method:fn(&Node, u32) -> f32, act_timer_ms:u64, nb_MCTS_simulations_max:u32, clean_mcts:bool) -> Agent {
	//initialize Agent from env and set default values
    let root_id = env.generate_id();
    let game_root = Node {game_state:env.clone(),
                            id:root_id.to_string(),
                            edges_out: Vec::with_capacity(9),
                            };
    
    let mcts = MCTS {root_id: root_id.to_string(), //create a copy because root id will be consume in hashmap
                        node_map: {
							let mut map = HashMap::with_capacity(TREE_CAPACITY);
							map.insert(root_id, game_root);
							map
						},
                        cpuct: cpuct,
                        };
    //create Agent
	let agent = Agent {mcts: mcts,
						nb_simulations:nb_MCTS_simulations_max,
						nb_evaluation_sim:nb_evaluation_sim,
						evaluation_method:evaluation_method,
						act_timer_ms:act_timer_ms,
						clean_mcts:clean_mcts,
					};
	return agent
}

//convert x,y moves (9x9 grid) into gn moves (9grids of 9cases)
fn xy_to_gn(x:u8, y:u8) -> (u8, u8) {
	let (g,n) = (x/3+y/3*3, x%3 + y%3*3);
	return (g,n)
}

fn gn_to_xy(gn:(u8, u8)) -> (u8, u8) {
	let (x,y) = (gn.0%3*3+gn.1%3, gn.0/3*3 +gn.1/3);
	return (x,y)
}

fn min_max(min_max:(i8,i8),value:i8) -> (i8,i8) {
	if value > min_max.1 {return (min_max.0,value)}
	else if value < min_max.0 {return (value, min_max.1)}
	else {return min_max}
}

fn min(a:f32,b:f32)->f32 {
	if a<b {return a}
	else {return b}
}

macro_rules! min_max {
	($x:expr) => ($x,$x);
	($x:expr, $($xs:expr),+) => {
		{
		let mut temp_values = ($x, $x);
		$(
			temp_values = min_max(temp_values, $xs);
		)+
		temp_values
		}
	};
}

fn board_value(board:&[f32;9]) -> f32 {
	//return board value as sum of win possibilities
	//sub calculation used for subboard and overboard
	//input board must be in range [0-1]
	//return is 1 capped
	let value = (board[0]*board[1]*board[2] + 
						board[3]*board[4]*board[5] +
						board[6]*board[7]*board[8] +
						board[0]*board[3]*board[6] +
						board[1]*board[4]*board[7] +
						board[2]*board[5]*board[8] +
						board[0]*board[4]*board[8] +
						board[2]*board[4]*board[6] );
	
	return min(1.0, value)
}

fn subboard_value(board:[i8;9]) -> (f32,f32) {
	//return (value) as (value_pA, value_pB)
	
	let mut boardA = [0.0;9];
	let mut boardB = [0.0;9];
	
	//parameters
	let empty_value = 0.3;	// value for OnGoing case
	
	for i in 0..9 {
		match board[i] {
			1 => {boardA[i] = 1.0; boardB[i] = 0.0;},
			-1 => {boardA[i] = 0.0; boardB[i] = 1.0;},
			0 => {boardA[i] = empty_value; boardB[i] = empty_value;},
			_ => panic!("invalid value in board"), //impossible
		}
	}
	let value = (board_value(&boardA), board_value(&boardB));
	
	return value
}

fn subboard_prior(board:[i8;9], board_idx:u8) -> Vec<((u8,u8),f32,f32)> {
	//expected gain for each player, each action on a 3x3 sub-board
	//return (prior) as array of ((board_idx, action),(prior_pB, prior_pA)) in range min_scale/1
	
	let mut moves:Vec<((u8,u8),f32,f32)> = Vec::with_capacity(9);
	let mut boardA = [0.0;9];
	let mut boardB = [0.0;9];
	
	//parameters
	let empty_value = 0.3;	// value for OnGoing case
	let scale_oppponent = 0.1; // weight of opponent move value
	let min_scale = 0.1;	//minimum prior
	
	for i in 0..9 {
		match board[i] {
			1 => {boardA[i] = 1.0; boardB[i] = 0.0;},
			-1 => {boardA[i] = 0.0; boardB[i] = 1.0;},
			0 => {boardA[i] = empty_value; boardB[i] = empty_value;},
			_ => panic!("invalid value in board"), //impossible
		}
	}
	
	if board[0]==0 {
		let value_a = min(1.0,boardA[1]*boardA[2] + boardA[3]*boardA[6] + boardA[4]*boardA[8]);
		let value_b = min(1.0,boardB[1]*boardB[2] + boardB[3]*boardB[6] + boardB[4]*boardB[8]);
		moves.push(((board_idx, 0u8),
					(value_a+scale_oppponent*value_b+min_scale)/(1.0+scale_oppponent+min_scale),
					(value_b+scale_oppponent*value_a+min_scale)/(1.0+scale_oppponent+min_scale),))
	}
	if board[1]==0 {
		let value_a = min(1.0,boardA[0]*boardA[2] + boardA[4]*boardA[7]);
		let value_b = min(1.0,boardB[0]*boardB[2] + boardB[4]*boardB[7]);
		moves.push(((board_idx, 1u8),
					(value_a+scale_oppponent*value_b+min_scale)/(1.0+scale_oppponent+min_scale),
					(value_b+scale_oppponent*value_a+min_scale)/(1.0+scale_oppponent+min_scale),))
	}
	if board[2]==0 {
		let value_a = min(1.0,boardA[0]*boardA[1] + boardA[5]*boardA[8] + boardA[4]*boardA[6]);
		let value_b = min(1.0,boardB[0]*boardB[1] + boardB[5]*boardB[8] + boardB[4]*boardB[6]);
		moves.push(((board_idx, 2u8),
					(value_a+scale_oppponent*value_b+min_scale)/(1.0+scale_oppponent+min_scale),
					(value_b+scale_oppponent*value_a+min_scale)/(1.0+scale_oppponent+min_scale),))
	}
	if board[3]==0 {
		let value_a = min(1.0,boardA[0]*boardA[6] + boardA[4]*boardA[5]);
		let value_b = min(1.0,boardB[0]*boardB[6] + boardB[4]*boardB[5]);
		moves.push(((board_idx, 3u8),
					(value_a+scale_oppponent*value_b+min_scale)/(1.0+scale_oppponent+min_scale),
					(value_b+scale_oppponent*value_a+min_scale)/(1.0+scale_oppponent+min_scale),))
	}
	if board[4]==0 {
		let value_a = min(1.0,boardA[3]*boardA[5] + boardA[1]*boardA[7] + boardA[0]*boardA[8] + boardA[2]*boardA[6]);
		let value_b = min(1.0,boardB[3]*boardB[5] + boardB[1]*boardB[7] + boardB[0]*boardB[8] + boardB[2]*boardB[6]);
		moves.push(((board_idx, 4u8),
					(value_a+scale_oppponent*value_b+min_scale)/(1.0+scale_oppponent+min_scale),
					(value_b+scale_oppponent*value_a+min_scale)/(1.0+scale_oppponent+min_scale),))
	}
	if board[5]==0 {
		let value_a = min(1.0,boardA[2]*boardA[8] + boardA[3]*boardA[4]);
		let value_b = min(1.0,boardB[2]*boardB[8] + boardB[3]*boardB[4]);
		moves.push(((board_idx, 5u8),
					(value_a+scale_oppponent*value_b+min_scale)/(1.0+scale_oppponent+min_scale),
					(value_b+scale_oppponent*value_a+min_scale)/(1.0+scale_oppponent+min_scale),))
	}
	if board[6]==0 {
		let value_a = min(1.0,boardA[0]*boardA[3] + boardA[7]*boardA[8] + boardA[4]*boardA[2]);
		let value_b = min(1.0,boardB[0]*boardB[3] + boardB[7]*boardB[8] + boardB[4]*boardB[2]);
		moves.push(((board_idx, 6u8),
					(value_a+scale_oppponent*value_b+min_scale)/(1.0+scale_oppponent+min_scale),
					(value_b+scale_oppponent*value_a+min_scale)/(1.0+scale_oppponent+min_scale),))
	}
	if board[7]==0 {
		let value_a = min(1.0,boardA[1]*boardA[4] + boardA[6]*boardA[8]);
		let value_b = min(1.0,boardB[1]*boardB[4] + boardB[6]*boardB[8]);
		moves.push(((board_idx, 7u8),
					(value_a+scale_oppponent*value_b+min_scale)/(1.0+scale_oppponent+min_scale),
					(value_b+scale_oppponent*value_a+min_scale)/(1.0+scale_oppponent+min_scale),))
	}
	if board[8]==0 {
		let value_a = min(1.0,boardA[0]*boardA[4] + boardA[2]*boardA[5] + boardA[6]*boardA[7]);
		let value_b = min(1.0,boardB[0]*boardB[4] + boardB[2]*boardB[5] + boardB[6]*boardB[7]);
		moves.push(((board_idx, 8u8),
					(value_a+scale_oppponent*value_b+min_scale)/(1.0+scale_oppponent+min_scale),
					(value_b+scale_oppponent*value_a+min_scale)/(1.0+scale_oppponent+min_scale),))
	}
	return moves
}

fn action_values(env:&TttEnv)-> Vec<((u8,u8),f32,f32)> {
	//return action, _value, prior for all available moves in env
	//prior is calculated as combinaison of direct move value and overboard evaluation
	//this function is not able to give an estimated value for the move, only which moves may be the best
	
	let mut moves :Vec<((u8,u8),f32,f32)> = Vec::new();
	match env.last_move {
		None => moves.extend((0..81).map(|x| ((x/9,x%9),1.0,1.0))),
		Some(gn) => {
			if env.over_board[gn.1 as usize] == GameStatus::OnGoing {
				moves.append(&mut subboard_prior(env.board[gn.1 as usize], gn.1));
			}
			else {
				for (idx_b, b) in env.board.iter().enumerate() {
					if env.over_board[idx_b] == GameStatus::OnGoing {
						moves.append(&mut subboard_prior(*b, idx_b as u8));
					}
				}
			}
		},
	}
	//add malus based on overboard status
	let overboard_mask = over_board_prior(env); //range ([0-2], [0-2])
	
	for (action,value_a,value_b) in moves.iter_mut() {
		// move_Gn_value = [(move_n_value @subgrid_G @player) - (overboard_n_value @opponent) + scaling ]* overboard_G_value @player
		// move value =    [^raw move value                   - ^opponent next move value     + scaling ]*  ^subgrid overall value
		//                 [[0.1;1]+[0;2]]*[1;3] = [0.1;9]   ... 
		if env.player_turn == 1 { //player_turn: A
			*value_a = (*value_a-overboard_mask[action.1 as usize].0 + 2.0) * (overboard_mask[action.0 as usize].0 + 1.0) / 9.0;
		}
		else { //player_turn: B
			*value_a = (*value_b-overboard_mask[action.1 as usize].1 + 2.0) * (overboard_mask[action.0 as usize].1 + 1.0) / 9.0;
		}
		*value_b = *value_a;
	}
	return moves
}

fn over_board_prior(env:&TttEnv)-> [(f32,f32);9] {
	//return overboard grid prior for pA, pB
	
	//parameters
	let malus = 2.0; //max value is 1+scale_oppponent
	let scale_oppponent = 0.1; // weight of opponent move value
	
	let mut overboard_prior = [(0.0,0.0);9];
	//create table from over_board with: current_player: 1, OnGoing:(subgrid evalutation), other: -1
	let mut overboard_a = [0.0;9];
	let mut overboard_b = [0.0;9];
	
	for i in 0..9 {
		match env.over_board[i] {
			GameStatus::Draw => (), //everyone stay at 0.0
			GameStatus::OnGoing => {
				//replace with subboard evaluation
				let value = subboard_value(env.board[i]);
				overboard_a[i] = value.0;
				overboard_b[i] = value.1;
			},
			GameStatus::Won(player) => {
				if player==1 {overboard_a[i] = 1.0;}	//WIN PLAYER A, B stay at 0.0
				else {overboard_b[i] = 1.0;}	//WIN PLAYER B, A stay at 0.0
			},
		}
	}
	
	if env.over_board[0]==GameStatus::OnGoing {
		let value_a = min(1.0,overboard_a[1]*overboard_a[2] + overboard_a[3]*overboard_a[6] + overboard_a[4]*overboard_a[8]); //max value=1, min=0
		let value_b = min(1.0,overboard_b[1]*overboard_b[2] + overboard_b[3]*overboard_b[6] + overboard_b[4]*overboard_b[8]);
		overboard_prior[0] = ((value_a+value_b*scale_oppponent), (value_b+value_a*scale_oppponent));
	}else {overboard_prior[0]=(malus,malus);}
	if env.over_board[1]==GameStatus::OnGoing {
		let value_a = min(1.0,overboard_a[0]*overboard_a[2] + overboard_a[4]*overboard_a[7]);
		let value_b = min(1.0,overboard_b[0]*overboard_b[2] + overboard_b[4]*overboard_b[7]);
		overboard_prior[1] = ((value_a+value_b*scale_oppponent), (value_b+value_a*scale_oppponent));
	}else {overboard_prior[0]=(malus,malus);}
	if env.over_board[2]==GameStatus::OnGoing {
		let value_a = min(1.0,overboard_a[0]*overboard_a[1] + overboard_a[4]*overboard_a[6] + overboard_a[5]*overboard_a[8]);
		let value_b = min(1.0,overboard_b[0]*overboard_b[1] + overboard_b[4]*overboard_b[6] + overboard_b[5]*overboard_b[8]);
		overboard_prior[2] = ((value_a+value_b*scale_oppponent), (value_b+value_a*scale_oppponent));
	}else {overboard_prior[0]=(malus,malus);}
	if env.over_board[3]==GameStatus::OnGoing {
		let value_a = min(1.0,overboard_a[0]*overboard_a[6] + overboard_a[4]*overboard_a[5]);
		let value_b = min(1.0,overboard_b[0]*overboard_b[6] + overboard_b[4]*overboard_b[5]);
		overboard_prior[3] = ((value_a+value_b*scale_oppponent), (value_b+value_a*scale_oppponent));
	}else {overboard_prior[0]=(malus,malus);}
	if env.over_board[4]==GameStatus::OnGoing {
		let value_a = min(1.0,overboard_a[0]*overboard_a[8] + overboard_a[1]*overboard_a[7] + overboard_a[2]*overboard_a[6] + overboard_a[3]*overboard_a[5]);
		let value_b = min(1.0,overboard_b[0]*overboard_b[8] + overboard_b[1]*overboard_b[7] + overboard_b[2]*overboard_b[6] + overboard_b[3]*overboard_b[5]);
		overboard_prior[4] = ((value_a+value_b*scale_oppponent), (value_b+value_a*scale_oppponent));
	}else {overboard_prior[0]=(malus,malus);}
	if env.over_board[5]==GameStatus::OnGoing {
		let value_a = min(1.0,overboard_a[2]*overboard_a[8] + overboard_a[3]*overboard_a[4]);
		let value_b = min(1.0,overboard_b[2]*overboard_b[8] + overboard_b[3]*overboard_b[4]);
		overboard_prior[5] = ((value_a+value_b*scale_oppponent), (value_b+value_a*scale_oppponent));
	}else {overboard_prior[0]=(malus,malus);}
	if env.over_board[6]==GameStatus::OnGoing {
		let value_a = min(1.0,overboard_a[0]*overboard_a[3] + overboard_a[2]*overboard_a[4] + overboard_a[7]*overboard_a[8]);
		let value_b = min(1.0,overboard_b[0]*overboard_b[3] + overboard_b[2]*overboard_b[4] + overboard_b[7]*overboard_b[8]);
		overboard_prior[6] = ((value_a+value_b*scale_oppponent), (value_b+value_a*scale_oppponent));
	}else {overboard_prior[0]=(malus,malus);}
	if env.over_board[7]==GameStatus::OnGoing {
		let value_a = min(1.0,overboard_a[1]*overboard_a[4] + overboard_a[6]*overboard_a[8]);
		let value_b = min(1.0,overboard_b[1]*overboard_b[4] + overboard_b[6]*overboard_b[8]);
		overboard_prior[7] = ((value_a+value_b*scale_oppponent), (value_b+value_a*scale_oppponent));
	}else {overboard_prior[0]=(malus,malus);}
	if env.over_board[8]==GameStatus::OnGoing {
		let value_a = min(1.0,overboard_a[0]*overboard_a[4] + overboard_a[2]*overboard_a[5] + overboard_a[6]*overboard_a[7]);
		let value_b = min(1.0,overboard_b[0]*overboard_b[4] + overboard_b[2]*overboard_b[5] + overboard_b[6]*overboard_b[7]);
		overboard_prior[8] = ((value_a+value_b*scale_oppponent), (value_b+value_a*scale_oppponent));
	}else {overboard_prior[0]=(malus,malus);}
	
	return overboard_prior
}

fn over_board_value(env:&TttEnv)-> (f32,f32) {
	//return overboard evalutation value for pA, pB
	
	//create table from over_board with: current_player: 1, OnGoing:(subgrid evalutation), other: -1
	let mut overboard_a = [0.0;9];
	let mut overboard_b = [0.0;9];
	
	for i in 0..9 {
		match env.over_board[i] {
			GameStatus::Draw => (), //everyone stay at 0.0
			GameStatus::OnGoing => {
				//replace with subboard evaluation
				let value = subboard_value(env.board[i]);
				overboard_a[i] = value.0;
				overboard_b[i] = value.1;
			},
			GameStatus::Won(player) => {
				if player==1 {overboard_a[i] = 1.0;}	//WIN PLAYER A, B stay at 0.0
				else {overboard_b[i] = 1.0;}	//WIN PLAYER B, A stay at 0.0
			},
		}
	}
	//calculate overboard value for pA, pB:
	let overboard_value = (board_value(&overboard_a), board_value(&overboard_b));
	
	return overboard_value
}

fn select_move(moves:Vec<((u8,u8),f32,f32)>) -> (u8,u8) {
	//return random move from bests moves availables
	//eprintln!("{:?}", moves);
	
	//selection weighted on prior
	let mut rng = rand::thread_rng();
	let action = moves.choose_weighted(&mut rng, |item| item.2).unwrap();
	
	return action.0
}

fn action_values_evaluation(node:&Node, nb_evaluation_sim:u32) -> f32 {
	//evaluation_method
	//evaluation: move selection based on action_values and scaled with game length
	let mut value = 0.0;
	for _i in 0..nb_evaluation_sim {
		//restore test_env
		let mut test_env = node.game_state.clone(); //as TttEnv
		let mut game_length = 0.0;
		
		//play until game is finish
		while test_env.status == GameStatus::OnGoing {
			let moves = action_values(&test_env); //get list of availaible actions
			let action = select_move(moves); //pick randomly one of the best available action => replace with random selection weighted on prior
			test_env.step(action);
			game_length += 1.0;
		}
		//update value for this round
		match test_env.status {
			GameStatus::Won(player) => {if player==node.game_state.player_turn {value += WIN/game_length;} else {value += LOSE/game_length;}},
			GameStatus::Draw => value += DRAW/game_length,
			GameStatus::OnGoing => panic!("game isn't over after simulation"),
		}
	}
	return value/(nb_evaluation_sim as f32).sqrt() 
}

fn short_action_values_evaluation(node:&Node, nb_evaluation_sim:u32) -> f32 {
	//evaluation_method
	//evaluation: move selection based on action_values with limited game deep then overboard evaluation
	let max_deep = 3;
	let mut value = 0.0;
	for _i in 0..nb_evaluation_sim {
		//restore test_env
		let mut test_env = node.game_state.clone(); //as TttEnv
		let mut game_length = 0;
		
		//play until game is finish or max deep reach
		while (test_env.status == GameStatus::OnGoing) & (game_length < max_deep) {
			let moves = action_values(&test_env); //get list of availaible actions
			let action = select_move(moves); //pick randomly one of the best available action => replace with random selection weighted on prior
			test_env.step(action);
			game_length += 1;
		}
		//update value for this round
		match test_env.status {
			GameStatus::Won(player) => {if player==node.game_state.player_turn {value += WIN;} else {value += LOSE;}},
			GameStatus::Draw => value += DRAW,
			GameStatus::OnGoing => {value += overboard_evaluation(node,1)},
		}
	}
	return value/(nb_evaluation_sim as f32).sqrt() 
}

fn overboard_evaluation(node:&Node, nb_evaluation_sim:u32) -> f32 {
	//evaluation_method
	//evaluation: overboard evaluation
	
	//restore test_env
	//let mut test_env = node.game_state.clone();
	
	let ob_value = over_board_value(&node.game_state);
	
	if node.game_state.player_turn == 1 {return ob_value.0-0.9*ob_value.1}
	else {return ob_value.1-0.9*ob_value.0}
}

fn battle_agent(agent1: &mut Agent, agent2: &mut Agent, nb_games:i32) -> [i32;3] {
	//use to compare agent with baseline
	//return WinA,WinB,Draw
	//initialize score
	let mut score = [0,0,0];
	let mut rng = rand::thread_rng();
	for i in 0..nb_games {
		println!("Game: {}", i);
		//initialize new game
		let mut env = TttEnv {..Default::default()};
		//reset agents mcts
		agent1.reset_mcts(&env);
		agent2.reset_mcts(&env);
		//chose playerA/playerB
		let first_agent1: bool = rng.gen();
		let mut current_agent1 = first_agent1;
		
		while env.status == GameStatus::OnGoing {
			let mut action = (0u8,0u8);
			let mut mcts_value = 0.0;
			let mut estimated_value = 0.0;
			if current_agent1 {
				(action,mcts_value,estimated_value) = agent1.act(&env);
				eprintln!("Agent1- action: {:?}, mcts_value: {:.4}, est_value: {:.4}", action, mcts_value, estimated_value);
				//eprintln!("mcts nodes Agent1: {}", agent1.mcts.node_map.len());
			}
			else {
			    (action,mcts_value,estimated_value) = agent2.act(&env);
			    eprintln!("Agent2- action: {:?}, mcts_value: {:.4}, est_value: {:.4}", action, mcts_value, estimated_value);
			}
			
			env.step(action);
			
			//switch agent
			current_agent1 = !current_agent1;
		}
		match env.status {
			GameStatus::Won(player) => {
				if (first_agent1 & (player==1))|(!first_agent1 & (player==-1)) {
				    score[0] += 1;
				    eprintln!("End of game, Agent1 Won");
				}
				else {
				    score[1] += 1;
				    eprintln!("End of game, Agent2 Won");
				}
			},
			GameStatus::Draw => {
			    score[2] += 1;
			    eprintln!("End of game, Draw");
			},
			GameStatus::OnGoing => panic!("game isn't over after simulation"),
		}
	}
	return score
}

fn timer_agent(agent: &mut Agent, env:&TttEnv, nb_simu:u32)->() {
	let mut total_time = time::Duration::from_secs(0);
	for i in 0..nb_simu {
		let start_time = time::Instant::now();
		let (action,value,_) = agent.act(&env);
		let elapsed_time = start_time.elapsed();
		println!("Simu: {}, {:?},{},{:?}", i, action, value,elapsed_time);
		total_time += elapsed_time;
		println!("mcts nodes: {}", agent.mcts.node_map.len());
	}
	println!("Elapsed time average: {:?}", total_time/nb_simu);
	
}

//For each allowed sub-board
//if ob[gn.1]==ongoing {[b[gn.1]]} else {[b[x] if ob[x]==ongoing]}
//get sub-board values for current player
//

macro_rules! parse_input {
    ($x:expr, $t:ident) => ($x.trim().parse::<$t>().unwrap())
}


/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/
 
static WIN: f32 = 1.0;
static DRAW: f32 = 0.5;
static LOSE: f32 = -1.0;
//static CLEAN_MCTS: bool = true;
static TREE_CAPACITY: usize = 1000000;
static ESTIMATE_VALUE: bool = true;

fn main() {
	//initialize game
	let mut env = TttEnv {..Default::default()};
	
	
	
	let mut agent = new_Agent(&env, 0.85, 1, overboard_evaluation, 50, 1000, false);//env, cpuct, nb_evaluation_sim, evaluation_method, act_timer_ms, nb_MCTS_simulations_max, clean_mcts
	let mut agent2 = new_Agent(&env, 0.85, 15, short_action_values_evaluation, 50, 30000, false);//env, cpuct, nb_evaluation_sim, evaluation_method, act_timer_ms, nb_MCTS_simulations_max, clean_mcts
	
	/*
	//set env to specific state
	let board = [[0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 0, 1, -1, 0, 0], 
				[-1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, -1, 0, -1], [0, 0, 0, 1, 0, -1, 0, 0, 0], 
				[0, 1, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, -1, 0, 0], [0, 0, -1, 0, 1, 0, 0, 0, 0]];
	let last_move = Some((2, 6));
	let player_turn = 1;
	env.set_state(board, last_move, player_turn);
	
	let (action, mcts_value, estimated_value) = agent.act(&env);
	eprintln!("action:{:?}, mcts value:{:?}, estimated value: {:?}", action, mcts_value, estimated_value);
	
	let root = agent.mcts.node_map.get(&agent.mcts.root_id).unwrap();
	eprintln!("root :{:?}", root);
	
	let ob_value = over_board_value(&env);
	eprintln!("overboard_value: {:?}", ob_value);
	
	//show evaluation value for this state (reproductible and coherent?)
	let node_id = agent2.mcts.add_node(&env);
	let node = agent2.mcts.node_map.get(&node_id).unwrap();
	for _i in 0..1 {
		let value = (agent2.evaluation_method)(node, 20);
		eprintln!("node value:{:?}",value);
		
	}
	*/
	
	//analyze act Duration
	//timer_agent(&mut agent, &env, 200);
	
	
	//fine tuning
	let score = battle_agent(&mut agent, &mut agent2, 200);
	println!("score:{:?}", score);
	//score:[33, 51, 16]
	
	//test results
	//score:[62, 109, 29]
	//0.7, 20, overboard_evaluation vs 0.7, 20, action_values_evaluation
	//update overboard_evaluation : score:[173, 19, 8]
	//
	//overboard_evaluation vs short_action_values_evaluation
	//score:[93, 36, 71]
	//score:[95, 49, 56] //clean mcts vs no cleaning
	//score:[47, 78, 75] //reduce deep to3, nb eval to 10
	
	//short_action_values_evaluation self
	//fine tuning
	//cpuct 0.7 vs 0.85[x]
	//cpuct 0.85[x] vs 0.95
	//nb simulations evaluation
	// 5[x] vs 10; 5[x] vs 15 (slight diff)
}