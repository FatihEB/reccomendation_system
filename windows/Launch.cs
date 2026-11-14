using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;

class Launcher
{
    static int Main()
    {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        try
        {
            string script;
            using (var reader = new StreamReader(Assembly.GetExecutingAssembly().GetManifestResourceStream("launch.ps1")))
                script = reader.ReadToEnd();
            using (var process = Process.Start(new ProcessStartInfo {
                FileName = Path.Combine(Environment.SystemDirectory, @"WindowsPowerShell\v1.0\powershell.exe"),
                Arguments = "-NoProfile -EncodedCommand " + Convert.ToBase64String(Encoding.Unicode.GetBytes(script)),
                WorkingDirectory = root,
                UseShellExecute = false
            }))
            {
                process.WaitForExit();
                if (process.ExitCode == 0) return 0;
                Console.Error.WriteLine("Setup exited with code " + process.ExitCode + ".");
            }
        }
        catch (Exception error) { Console.Error.WriteLine(error.Message); }
        Console.Error.WriteLine("Press Enter to close.");
        Console.ReadLine();
        return 1;
    }
}
