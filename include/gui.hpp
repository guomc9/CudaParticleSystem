#ifndef __GUI_H__
#define __GUI_H__
#include <iostream>
#include "context.h"

class Gui
{
    private:
        int _fps;
        int _simulation_steps;
        int _current_step;
        int _height;
        int _width;
        float _delta_time;
        float _eye_pos[3];
        float _lookat[3];
        float _up[3];
        float _move_speed;
        float _rot_speed;

        inline void _rotateAroundYAxis(const float angle)
        {
            float3 eye_pos = make_float3(_eye_pos[0], _eye_pos[1], _eye_pos[2]);
            float3 lookat = make_float3(_lookat[0], _lookat[1], _lookat[2]);
            float3 dir = sub_f3(eye_pos, lookat);
            dir = rotate_around_y(dir, angle);
            eye_pos = sub_f3(lookat, dir);
            _eye_pos[0] = eye_pos.x;
            _eye_pos[1] = eye_pos.y;
            _eye_pos[2] = eye_pos.z;
        }

        inline void _rotateAroundXAxis(const float angle)
        {
            float3 eye_pos = make_float3(_eye_pos[0], _eye_pos[1], _eye_pos[2]);
            float3 lookat = make_float3(_lookat[0], _lookat[1], _lookat[2]);
            float3 up = make_float3(_up[0], _up[1], _up[2]);
            float3 dir = sub_f3(eye_pos, lookat);
            dir = rotate_around_x(dir, angle);
            eye_pos = sub_f3(lookat, dir);
            up = rotate_around_x(up, angle);
            _eye_pos[0] = eye_pos.x;
            _eye_pos[1] = eye_pos.y;
            _eye_pos[2] = eye_pos.z;
            _up[0] = up.x;
            _up[1] = up.y;
            _up[2] = up.z;
        }

    public:
        Gui(int height, int width, float3 eye_pos, float3 lookat, float3 up)
        :_height(height), _width(width), _delta_time(0.001f), _simulation_steps(5), _current_step(0), _fps(5), _move_speed(0.01f), _rot_speed(0.01f)
        {
            _eye_pos[0] = eye_pos.x;
            _eye_pos[1] = eye_pos.y;
            _eye_pos[2] = eye_pos.z;
            _lookat[0] = lookat.x;
            _lookat[1] = lookat.y;
            _lookat[2] = lookat.z;
            _up[0] = up.x;
            _up[1] = up.y;
            _up[2] = up.z;

        };

        int* fps()
        {
            return &_fps;
        }

        float* move_speed()
        {
            return &_move_speed;
        }

        float* rot_speed()
        {
            return &_rot_speed;
        }

        int* height()
        {
            return &_height;
        }

        int* width()
        {
            return &_width;
        }

        float* eye_pos()
        {
            return _eye_pos;
        }

        float* lookat()
        {
            return _lookat;
        }

        float* up()
        {
            return _up;
        }

        float* delta_time()
        {
            return &_delta_time;
        }

        int* simulation_steps()
        {
            return &_simulation_steps;
        }

        int* current_step()
        {
            return &_current_step;
        }

        inline float3 get_eye_pos_vec() const
        {
            return make_float3(_eye_pos[0], _eye_pos[1], _eye_pos[2]);
        }

        inline float3 get_lookat_vec() const
        {
            return make_float3(_lookat[0], _lookat[1], _lookat[2]);
        }

        inline float3 get_up_vec() const
        {
            return make_float3(_up[0], _up[1], _up[2]);
        }

        inline int get_height() const
        {
            return _height;
        }

        inline int get_width() const
        {
            return _width;
        }

        inline float get_delta_time() const
        {
            return _delta_time;
        }

        inline int get_simulation_steps() const
        {
            return _simulation_steps;
        }

        inline int get_current_step() const
        {
            return _current_step;
        }

        inline int get_fps() const
        {
            return _fps;
        }

        inline float get_move_speed() const
        {
            return _move_speed;
        }

        inline float get_rot_speed() const
        {
            return _rot_speed;
        }

        inline void update(GLFWwindow* window)
        {
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            {
                _eye_pos[2] -= _move_speed;
                _lookat[2] -= _move_speed;
            }
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            {
                _eye_pos[2] += _move_speed;
                _lookat[2] += _move_speed;
            }
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            {
                _eye_pos[0] += _move_speed;
                _lookat[0] += _move_speed;
            }
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            {
                _eye_pos[0] -= _move_speed;
                _lookat[0] -= _move_speed;
            }
            if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS)
            {
                _rotateAroundYAxis(_rot_speed);
            }
            if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS)
            {
                _rotateAroundYAxis(-_rot_speed);
            }
            if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
            {
                _rotateAroundXAxis(_rot_speed);

            }
            if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
            {
                _rotateAroundXAxis(-_rot_speed);
            }
        }

};
#endif