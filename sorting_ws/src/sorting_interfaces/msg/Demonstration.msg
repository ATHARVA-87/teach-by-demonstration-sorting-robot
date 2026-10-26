# One completed teaching example: "this item goes in this box".
# Only the goal is recorded, never the hand trajectory.

std_msgs/Header header   # stamp = time of the drop
string session_id        # changes with every new scene; listeners reset when it changes
int32 scene_seed         # seed of the sorting_scene scene being taught on
uint32 index             # 1-based order within the session
Item item                # attributes and start position of the item
string box               # label of the target box, e.g. "A"
string source            # "ar" (worker demo) | "query" (answer to an active-learning question) | "simulated_teacher"
float64 duration_s       # seconds from grab to drop (teaching-time metric)
